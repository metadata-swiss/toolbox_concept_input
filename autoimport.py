from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import pandas as pd
from datetime import datetime
import os
import re
import json
import base64
import hmac
import hashlib
import jwt
from pandas.api.types import is_numeric_dtype, is_object_dtype
import requests
import config
import uuid
import pickle
import unicodedata
import string
from collections import defaultdict
from langdetect import detect
import tempfile
import openai
import io
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create the Flask app first, before any circular imports might occur
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.secret_key = config.SECRET_KEY  # Use config value from environment


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# Add error handlers to return JSON instead of HTML for API requests
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum file size is 16MB.'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logging.exception(f"Internal server error on {request.path}: {error}")
    # Check if this is an API request (JSON expected)
    if request.path.startswith('/api/') or request.is_json or request.accept_mimetypes.accept_json:
        return jsonify({'error': 'Internal server error. Please try again or contact support.'}), 500
    # Otherwise return HTML error page
    return "Internal Server Error", 500

@app.errorhandler(Exception)
def handle_exception(error):
    """Handle any unhandled exceptions"""
    logging.exception(f"Unhandled exception on {request.path}: {error}")
    # Check if this is an API request or file upload
    if request.path.startswith('/api/') or request.is_json or request.accept_mimetypes.accept_json or request.path == '/':
        return jsonify({'error': 'An error occurred. Please try again or contact support.'}), 500
    # Otherwise re-raise to let Flask handle it
    raise error

# Create a directory to store session data files
SESSION_DATA_DIR = os.path.join(tempfile.gettempdir(), 'i14y-concept-import-sessions')
if not os.path.exists(SESSION_DATA_DIR):
    os.makedirs(SESSION_DATA_DIR)

# Session data cleanup functions
def cleanup_old_session_files():
    """Remove session files older than 24 hours"""
    try:
        current_time = time.time()
        max_age_seconds = 24 * 60 * 60  # 24 hours
        
        for filename in os.listdir(SESSION_DATA_DIR):
            if filename.endswith('.pkl'):
                file_path = os.path.join(SESSION_DATA_DIR, filename)
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        print(f"Cleaned up old session file: {filename}")
                except OSError as e:
                    print(f"Error removing session file {filename}: {e}")
    except Exception as e:
        print(f"Error during session cleanup: {e}")

def cleanup_session_file(session_id):
    """Remove a specific session file"""
    try:
        file_path = os.path.join(SESSION_DATA_DIR, f"{session_id}.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up session file for session: {session_id}")
    except Exception as e:
        print(f"Error removing session file for {session_id}: {e}")

def periodic_cleanup_worker():
    """Background worker that runs cleanup every hour"""
    while True:
        try:
            cleanup_old_session_files()
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")
        time.sleep(60 * 60)  # Sleep for 1 hour

# Start the periodic cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup_worker, daemon=True)
cleanup_thread.start()

# Helper functions to save and load session data
#
# Session files are signed with HMAC-SHA256 to prevent tampering. On disk each
# file has the layout:
#     [32 bytes: HMAC-SHA256(payload) using app.secret_key][pickle payload]
#
# pickle.load() executes code from the byte stream during deserialization, so an
# attacker who can write to SESSION_DATA_DIR could otherwise achieve RCE. The
# HMAC prefix ensures that only payloads produced by this application (which
# knows the secret key) are ever passed to pickle.load().

def _hmac_key():
    key = app.secret_key
    if isinstance(key, str):
        key = key.encode('utf-8')
    if not key:
        raise RuntimeError('Cannot sign session data: app.secret_key is not configured')
    return key


def _compute_session_mac(payload_bytes):
    return hmac.new(_hmac_key(), payload_bytes, hashlib.sha256).digest()


def save_session_data(data, session_id=None):
    """Save session data to a file and return the session identifier.

    The file is signed with HMAC-SHA256 so that load_session_data() can refuse
    to pickle.load() any bytes that were not produced by this process.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    file_path = os.path.join(SESSION_DATA_DIR, f"{session_id}.pkl")
    # Use atomic write: write to temp file first, then rename
    temp_path = file_path + '.tmp'
    try:
        payload = pickle.dumps(data)
        mac = _compute_session_mac(payload)
        with open(temp_path, 'wb') as f:
            f.write(mac)
            f.write(payload)
        # Atomic rename
        os.replace(temp_path, file_path)
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        print(f"Error saving session data: {e}")
        raise
    return session_id

def load_session_data(session_id):
    """Load session data from a file based on ID, verifying its HMAC signature."""
    file_path = os.path.join(SESSION_DATA_DIR, f"{session_id}.pkl")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        if len(raw) < 32:
            print(f"Session file {session_id} is too short to contain a valid signature")
            return None
        stored_mac = raw[:32]
        payload = raw[32:]
        expected_mac = _compute_session_mac(payload)
        if not hmac.compare_digest(stored_mac, expected_mac):
            # Signature mismatch: refuse to deserialize (potentially tampered file)
            print(f"Session file {session_id} failed signature verification; refusing to load")
            return None
        # Payload authenticity verified via HMAC above; safe to unpickle.
        return pickle.loads(payload)  # nosec B301
    except (EOFError, pickle.UnpicklingError) as e:
        # File is corrupted or empty
        print(f"Error loading session data {session_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading session data {session_id}: {e}")
        return None

def load_uploaded_dataframe(session_data):
    """Reconstruct the original dataframe from the stored upload."""
    dataset_info = session_data.get('dataset_file')
    if not dataset_info:
        return None
    if isinstance(dataset_info, dict):
        dataset_path = dataset_info.get('path')
        dataset_type = (dataset_info.get('type') or '').lower()
    else:
        dataset_path = dataset_info
        dataset_type = os.path.splitext(dataset_path)[1].lower()
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    if dataset_type in ('.xlsx', '.xls'):
        return pd.read_excel(dataset_path)
    if dataset_type == '.csv':
        with open(dataset_path, 'rb') as source:
            return read_csv_with_encoding_fallback(io.BytesIO(source.read()), sep=None, engine='python')
    raise ValueError(f"Unsupported dataset type: {dataset_type or 'unknown'}")

def infer_language_from_column(column_name):
    """Heuristically detect the language from a column suffix."""
    lowered = column_name.lower()
    for delimiter in ['_', '-', ' ']:
        if delimiter in lowered:
            suffix = lowered.rsplit(delimiter, 1)[-1]
            if suffix in {'de', 'fr', 'it', 'en', 'rm'}:
                return suffix
    if lowered.endswith(('de', 'fr', 'it', 'en', 'rm')):
        return lowered[-2:]
    return None

def safe_detect_language(text, default='de'):
    """Detect language safely with a fallback."""
    try:
        if not text or not text.strip():
            return default
        return detect(text)
    except Exception:
        return default

# Add theme definitions
VALID_THEMES = {
    "101": "Work",  # "Arbeit"
    "102": "Construction",  # "Bauen"
    "103": "Education",  # "Bildung"
    "104": "Foreign Relations",  # "Aussenbeziehungen"
    "105": "Jurisdiction",  # "Gerichtsbarkeit"
    "106": "Society",  # "Gesellschaft"
    "107": "Political Activities",  # "Politische Aktivitäten"
    "108": "Culture",  # "Kultur"
    "109": "Agriculture",  # "Landwirtschaft"
    "110": "Infrastructure",  # "Infrastruktur"
    "111": "Security",  # "Sicherheit"
    "112": "Taxes",  # "Steuern"
    "113": "Environment",  # "Umwelt"
    "114": "Health",  # "Gesundheit"
    "115": "Economy",  # "Wirtschaft"
    "116": "Mobility",  # "Mobilität"
    "117": "Residents",  # "Einwohner"
    "118": "Companies",  # "Unternehmen"
    "119": "Authorities",  # "Behörden"
    "120": "Buildings and Real Estate",  # "Gebäude und Grundstücke"
    "121": "Animals",  # "Tiere"
    "122": "Geoinformation",  # "Geoinformationen"
    "123": "Legal Framework",  # "Rechtssammlung"
    "124": "Energy",  # "Energie"
    "125": "Public Statistics",  # "Öffentliche Statistik"
    "126": "Social Security"  # "Soziale Sicherheit"
}

def simple_multilingual(text):
    """Create a simple multilingual object without performing translations"""
    return {
        "de": text,
        "en": text,
        "fr": text,
        "it": text,
        "rm": text,
    }

def generate_regex_pattern(values):
    """Generate a regex pattern that matches all strings in the given list using pure Python."""
    if not values:
        return ""
    
    # For numeric values, just return a generic pattern
    if all(isinstance(v, (int, float)) for v in values):
        max_digits = max(len(str(int(v))) for v in values)
        return f"\\d{{{max_digits}}}"
    
    # For strings, prepare valid samples
    samples = [str(v) for v in values[:50] if v is not None and str(v).strip()]  # Use up to 50 samples, skip empty values
    
    if not samples:
        return ".*"  # Default if no valid samples
    
    # Check for common patterns first
    date_formats = [
        # Standard ISO format: YYYY-MM-DD
        (r'^\d{4}-\d{2}-\d{2}$', "\\d{4}-\\d{2}-\\d{2}"),
        # European format: DD.MM.YYYY
        (r'^\d{2}\.\d{2}\.\d{4}$', "\\d{2}\\.\\d{2}\\.\\d{4}"),
        # European format with flexible day/month: D.M.YYYY or DD.MM.YYYY
        (r'^\d{1,2}\.\d{1,2}\.\d{4}$', "\\d{1,2}\\.\\d{1,2}\\.\\d{4}"),
        # US format: MM/DD/YYYY
        (r'^\d{2}/\d{2}/\d{4}$', "\\d{2}/\\d{2}/\\d{4}"),
        # Flexible slash format: M/D/YYYY or MM/DD/YYYY
        (r'^\d{1,2}/\d{1,2}/\d{4}$', "\\d{1,2}/\\d{1,2}/\\d{4}"),
        # Compact format: YYYYMMDD
        (r'^\d{8}$', "\\d{8}"),
        # ISO format with time: YYYY-MM-DD HH:MM:SS
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', "\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}")
    ]
    
    # More permissive date pattern detection - match if 80% of samples match the pattern
    for pattern, regex in date_formats:
        matching_count = sum(1 for s in samples if re.match(pattern, s))
        match_percentage = matching_count / len(samples)
        if match_percentage > 0.8:  # If 80% or more samples match
            return regex
    
    # Email pattern special case
    if all('@' in s for s in samples):
        return "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    
    # UUID pattern special case
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if all(re.match(uuid_pattern, s.lower()) for s in samples):
        return "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    
    # Fallback to simple patterns for specific cases
    if len(set(len(s) for s in samples)) == 1:
        sample_length = len(samples[0])
        
        # Check if all characters are numeric
        if all(s.isdigit() for s in samples):
            return f"\\d{{{sample_length}}}"
        
        # Check if all characters are alphabetic
        if all(s.isalpha() for s in samples):
            return f"[A-Za-z]{{{sample_length}}}"
            
        # Check if all characters are alphanumeric
        if all(s.isalnum() for s in samples):
            return f"[A-Za-z0-9]{{{sample_length}}}"
    
    # Check for common character patterns
    has_letters = any(c.isalpha() for s in samples for c in s)
    has_digits = any(c.isdigit() for s in samples for c in s)
    has_spaces = any(' ' in s for s in samples)
    has_special = any(not c.isalnum() and c != ' ' for s in samples for c in s)
    
    if has_letters and has_digits and not has_spaces and not has_special:
        # Alphanumeric without spaces or special chars
        min_length = min(len(s) for s in samples)
        max_length = max(len(s) for s in samples)
        if min_length == max_length:
            return f"[A-Za-z0-9]{{{min_length}}}"
        else:
            return f"[A-Za-z0-9]{{{min_length},{max_length}}}"
    
    # Last resort - return a generic pattern
    return ".*"

def analyze_column_type(series):
    # First check for all null values
    if series.isna().all():
        return {'type': 'String', 'is_codelist': False}
    
    column_name = series.name.lower() if hasattr(series, 'name') else ""
    
    # Special handling for year columns
    is_year_column = (
        'jahr' in column_name or 
        'year' in column_name or 
        'annee' in column_name or 
        'anno' in column_name
    )
    
    # Check for year values (4-digit numbers between 1900 and current year + 100)
    current_year = datetime.now().year
    def is_year_value(val):
        try:
            val = int(val)
            return 1900 <= val <= current_year + 100
        except:
            return False
    
    # Check if all values are years
    all_years = False
    if is_numeric_dtype(series) or is_object_dtype(series):
        all_years = all(is_year_value(val) for val in series.dropna())
    
    # If it's a year column by name or all values are years, treat as Number
    if (is_year_column or all_years) and len(series.dropna().unique()) <= 200:
        return {
            'type': 'Number',
            'is_codelist': False,
            'format': 'YYYY',
            'pattern': '\\d{4}'
        }
    
    # Check if datetime - use explicit formats to avoid warnings
    is_date = False
    date_format = None
    
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d',    # YYYY-MM-DD
        '%d.%m.%Y',    # DD.MM.YYYY
        '%d/%m/%Y',    # DD/MM/YYYY
        '%m/%d/%Y',    # MM/DD/YYYY
        '%Y%m%d',      # YYYYMMDD
        '%d-%m-%Y'     # DD-MM-YYYY
    ]
    
    sample_values = series.dropna().head(10).astype(str)
    if pd.api.types.is_datetime64_any_dtype(series):
        is_date = True
        date_format = infer_date_format(series)
    else:
        # Try each format explicitly
        for fmt in date_formats:
            try:
                # Try to parse with this format - specify errors='raise' to avoid silent conversion
                pd.to_datetime(sample_values, format=fmt, errors='raise')
                is_date = True
                date_format = fmt_to_readable(fmt)
                break
            except:
                continue
        
        # If none worked, check common patterns with regex instead of using dateutil
        if not is_date:
            date_patterns = [
                (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD'),
                (r'^\d{2}\.\d{2}\.\d{4}$', 'DD.MM.YYYY'),
                (r'^\d{2}/\d{2}/\d{4}$', 'DD/MM/YYYY')
            ]
            
            for pattern, fmt in date_patterns:
                if all(re.match(pattern, s) for s in sample_values):
                    is_date = True
                    date_format = fmt
                    break
    
    if is_date:
        pattern = generate_regex_pattern(series.dropna().tolist())
        # Ensure we have a proper date pattern even if generate_regex_pattern fails
        if pattern == ".*" and date_format:
            # Use format to determine pattern
            if date_format == 'YYYY-MM-DD':       
                pattern = "\\d{4}-\\d{2}-\\d{2}"
            elif date_format == 'DD.MM.YYYY':
                pattern = "\\d{2}\\.\\d{2}\\.\\d{4}"
            elif date_format == 'DD/MM/YYYY':
                pattern = "\\d{2}/\\d{2}/\\d{4}"
                
        return {
            'type': 'Date',
            'is_codelist': False,
            'format': date_format,
            'pattern': pattern
        }
    
    # Get basic statistics
    unique_values = series.dropna().unique()
    total_values = len(series.dropna())
    
    # Check if numeric
    is_numeric = False
    if pd.api.types.is_numeric_dtype(series):
        is_numeric = True
    else:
        try:
            pd.to_numeric(series, errors='raise')
            is_numeric = True
        except:
            pass
    
    # Criteria for codelist remains the same, but we'll exclude years and true numeric values
    is_codelist = (
        len(unique_values) < 50 and 
        total_values >= 5 and 
        len(unique_values) < total_values/2
    )
    
    # Additional checks for numeric values
    if is_numeric:
        # If values look like measurements (many different values, decimals), not a codelist
        values = pd.to_numeric(series.dropna())
        has_decimals = any(v % 1 != 0 for v in values)
        high_cardinality = len(unique_values) > 20
        
        if has_decimals or high_cardinality:
            is_codelist = False
        
        # For small integers, additional check if it looks like a code
        if not is_codelist and not has_decimals:
            if (len(unique_values) <= 10 and
                max(values) <= 100 and
                values.dtype in ['int64', 'int32']):
                is_codelist = True
    
    if is_codelist:
        value_counts = series.value_counts()
        
        # Calculate length statistics for codes
        if is_numeric:
            code_lengths = [len(str(val)) for val in unique_values]
            code_type = 'numeric'
        else:
            code_lengths = [len(str(val)) for val in unique_values]
            code_type = 'string'
        
        return {
            'type': 'Codelist',
            'is_codelist': True,
            'unique_values': sorted(unique_values.tolist())[:20],  # Limit unique values
            'value_counts': {str(k): v for k, v in list(value_counts.items())[:20]},  # Limit value counts
            'total_values': total_values,
            'unique_count': len(unique_values),
            'code_type': code_type,
            'min_length': min(code_lengths),
            'max_length': max(code_lengths)
        }
    
    # Make sure to include pattern in the Number type too
    if is_numeric and not is_codelist:
        # Generate pattern for numeric values
        values = pd.to_numeric(series.dropna())
        has_decimals = any(v % 1 != 0 for v in values)
        
        if has_decimals:
            # For floating point values
            max_decimals = max(len(str(v).split('.')[1]) if '.' in str(v) else 0 
                               for v in values)
            pattern = f"\\d+\\.\\d{{{max_decimals}}}"
        else:
            # For integer values
            max_digits = max(len(str(int(v))) for v in values)
            pattern = f"\\d{{{max_digits}}}"
            
        return {
            'type': 'Number',
            'is_codelist': False,
            'pattern': pattern
        }
    
    # For string values, generate a pattern
    if not is_numeric:
        pattern = generate_regex_pattern(series.dropna().tolist())
        return {
            'type': 'String',
            'is_codelist': False,
            'pattern': pattern
        }
    
    # Default to Number for numeric columns
    return {'type': 'Number', 'is_codelist': False}

def infer_date_format(series):
    """Infer the date format from a series."""
    # Convert to strings first
    samples = [str(val)[:10] for val in series.dropna().tolist()[:5]]
    
    # Check common patterns
    if all(re.match(r'^\d{4}-\d{2}-\d{2}$', s) for s in samples):
        return 'YYYY-MM-DD'
    if all(re.match(r'^\d{2}\.\d{2}\.\d{4}$', s) for s in samples):
        return 'DD.MM.YYYY'
    if all(re.match(r'^\d{2}/\d{2}/\d{4}$', s) for s in samples):
        return 'DD/MM/YYYY'
    
    # Default format
    return 'YYYY-MM-DD'

def fmt_to_readable(fmt):
    """Convert Python date format to human-readable format"""
    mapping = {
        '%Y-%m-%d': 'YYYY-MM-DD',
        '%d.%m.%Y': 'DD.MM.YYYY',
        '%d/%m/%Y': 'DD/MM/YYYY',
        '%m/%d/%Y': 'MM/DD/YYYY',
        '%Y%m%d': 'YYYYMMDD',
        '%d-%m-%Y': 'DD-MM-YYYY'
    }
    return mapping.get(fmt, fmt)

def fetch_user_agencies_from_api(token):
    """Fetch user's agencies from the I14Y API"""
    try:
        # Clean the token
        clean_token = token.strip()
        if clean_token.upper().startswith('BEARER '):
            clean_token = clean_token[7:].strip()
        
        # Call the I14Y API to get the user's agencies
        url = 'https://core.i14y.c.bfs.admin.ch/api/Agents/'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {clean_token}'
        }
        
        response = requests.get(url, headers=headers, timeout=(5, 30))
        
        if response.status_code == 200:
            agents_data = response.json()
            
            # Parse the response - expecting a list of agents
            agencies = []
            if isinstance(agents_data, list):
                for agent in agents_data:
                    # Extract identifier and name from the agent object
                    identifier = agent.get('identifier')
                    name = agent.get('name', {})
                    
                    if identifier:
                        # If name is not a multilingual object, convert it
                        if isinstance(name, str):
                            name = simple_multilingual(name)
                        
                        agencies.append({
                            'identifier': identifier,
                            'name': name
                        })
            
            # Sort agencies by German name for better UI
            if agencies:
                agencies.sort(key=lambda x: x.get('name', {}).get('de', '').lower())
                return agencies
        else:
            print(f"Failed to fetch agencies from API: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        logging.exception(f"Error fetching agencies: {e}")
    
    return []

def fetch_user_agencies(token):
    """Extract agencies from the JWT token"""
    if not token:
        return []
    
    # First try to fetch from the I14Y API (more authoritative)
    agencies = fetch_user_agencies_from_api(token)
    if agencies:
        return agencies
    
    # Fallback to extracting from JWT token
    try:
        # Remove "BEARER " or "Bearer " prefix if present (case-insensitive)
        jwt_token = token.strip()
        if jwt_token.upper().startswith('BEARER '):
            jwt_token = jwt_token[7:].strip()  # Remove "BEARER " (7 characters)
        
        # JWT tokens are in format: header.payload.signature
        token_parts = jwt_token.split('.')
        if len(token_parts) >= 2:
            # Decode the payload part
            payload = token_parts[1]
            payload += '=' * ((4 - len(payload) % 4) % 4)  # Add padding
            decoded_payload = base64.b64decode(payload).decode('utf-8')
            payload_data = json.loads(decoded_payload)
            
            # Extract agencies - they come as simple strings like "6517609\\i14y-test-organisation"
            agency_strings = payload_data.get('agencies', [])
            
            if agency_strings:
                # Convert agency strings to proper agency objects
                agencies = []
                for agency_str in agency_strings:
                    # Split by backslash to get identifier and name
                    # The format is: "identifier\\name" or just "identifier"
                    parts = agency_str.split('\\')
                    if len(parts) >= 2:
                        identifier = parts[0]
                        name = parts[1]
                    else:
                        identifier = agency_str
                        name = agency_str
                    
                    # Create agency object with multilingual name
                    agencies.append({
                        'identifier': identifier,
                        'name': simple_multilingual(name)
                    })
                
                # Sort agencies by German name for better UI
                agencies.sort(key=lambda x: x.get('name', {}).get('de', '').lower())
                return agencies
    except Exception as e:
        # Log the error for debugging
        logging.exception(f"Error fetching user agencies: {e}")
    
    return []



def read_csv_with_encoding_fallback(file, **kwargs):
    """Read CSV content trying multiple encodings and delimiters."""
    from io import StringIO, TextIOWrapper, BytesIO
    
    if hasattr(file, 'read'):
        pos = file.tell() if hasattr(file, 'tell') else None
        content = file.read()
        if hasattr(file, 'seek') and pos is not None:
            try:
                file.seek(pos)
            except Exception:
                pass
    else:
        content = file
    
    if isinstance(content, str):
        return pd.read_csv(StringIO(content), **kwargs)
    
    raw_bytes = content if isinstance(content, bytes) else str(content).encode('utf-8')
    
    detected_encoding = None
    try:
        import chardet  # type: ignore
        detected = chardet.detect(raw_bytes)
        if detected and detected.get('encoding'):
            detected_encoding = detected['encoding']
    except ImportError:
        pass
    
    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1', 'utf-16', 'utf-32']
    if detected_encoding:
        if detected_encoding in encodings:
            encodings.remove(detected_encoding)
        encodings.insert(0, detected_encoding)
    
    last_error = None
    for encoding in encodings:
        try:
            text_io = TextIOWrapper(BytesIO(raw_bytes), encoding=encoding, newline='')
            return pd.read_csv(text_io, **kwargs)
        except Exception as exc:
            last_error = exc
            continue
    
    raise ValueError(f"Could not decode or parse the CSV file. Last error: {last_error}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html', themes=VALID_THEMES)
    
    if request.method == 'POST':
        try:
            # No token required at upload time - it will be provided later
            
            # Now we don't need token or agency validation upfront
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            
            file_ext = os.path.splitext(file.filename)[1].lower()
            # Canonical mapping for allowed file types
            allowed_extensions = {'.xlsx': '.xlsx', '.xls': '.xls', '.csv': '.csv'}
            # Only accept one of the allowed extensions
            dataset_suffix = allowed_extensions.get(file_ext)
            file_bytes = file.read()
            if not file_bytes:
                return jsonify({'error': 'Uploaded file is empty'}), 400
            
            if dataset_suffix == '.xlsx' or dataset_suffix == '.xls':
                df = pd.read_excel(io.BytesIO(file_bytes))
            elif dataset_suffix == '.csv':
                df = read_csv_with_encoding_fallback(io.BytesIO(file_bytes), sep=None, engine='python')
            else:
                return jsonify({'error': 'Unsupported file type. Please upload a .xlsx, .xls, or .csv file'}), 400
        except ValueError as e:
            logging.exception(f"ValueError reading file: {e}")
            return jsonify({'error': 'Failed to read the file. Please check the file format and encoding.'}), 400
        except Exception as e:
            logging.exception(f"Exception reading file: {e}")
        
        session_id = str(uuid.uuid4())
        dataset_path = os.path.join(SESSION_DATA_DIR, f"{session_id}{dataset_suffix}")
        with open(dataset_path, 'wb') as dataset_file:
            dataset_file.write(file_bytes)

        columns_info = []
        form_data = {
            'theme': request.form.get('theme')
        }
        
        # Validate theme
        if not form_data['theme']:
            return jsonify({'error': 'Theme is required'})
        if form_data['theme'] not in VALID_THEMES:
            return jsonify({'error': 'Invalid theme code'})
        
        for column in df.columns:
            analysis = analyze_column_type(df[column])
            column_info = {
                'name': column,
                'type': analysis['type'],
                'is_codelist': analysis['is_codelist']
            }
            if analysis['is_codelist']:
                column_info.update({
                    'unique_values': analysis['unique_values'][:20],
                    'value_counts': {str(k): v for k, v in list(analysis['value_counts'].items())[:20]},
                    'code_type': analysis['code_type'],
                    'min_length': analysis['min_length'],
                    'max_length': analysis['max_length'],
                    'unique_count': analysis['unique_count']
                })
            if 'pattern' in analysis:
                column_info['pattern'] = analysis['pattern']
            if 'format' in analysis:
                column_info['format'] = analysis['format']
            columns_info.append(column_info)
        
        session_data = {
            'columns': columns_info,
            'form_data': form_data,
            'dataset_file': {
                'path': dataset_path,
                'type': dataset_suffix,
                'original_name': file.filename
            }
        }
        
        session_id = save_session_data(session_data, session_id=session_id)
        session['session_data_id'] = session_id
        
        return jsonify({
            'success': True,
            'redirect_url': url_for('results')
        })

    return render_template('index.html', themes=VALID_THEMES)

@app.route('/logout', methods=['POST'])
def logout():
    """Clean up session data and redirect to upload page"""
    if 'session_data_id' in session:
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
    return redirect(url_for('upload_file'))

@app.route('/api/get-default-description/<int:index>', methods=['GET'])
def get_default_description(index):
    """Generate a default description for a concept"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired'}), 404
    
    if index < 0 or index >= len(session_data.get('columns', [])):
        return jsonify({'error': 'Invalid column index'}), 400
    
    column = session_data['columns'][index]
    
    # Generate default description in German (default language)
    description = generate_default_description(column, lang='de')
    
    return jsonify({'description': description})

@app.route('/api/generate-description/<int:index>', methods=['POST'])
def generate_description(index):
    """Generate an AI-powered description for a concept"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired'}), 404
    
    if index < 0 or index >= len(session_data.get('columns', [])):
        return jsonify({'error': 'Invalid column index'}), 400
    
    column = session_data['columns'][index]
    
    # Get request data
    data = request.get_json() or {}
    title = data.get('title', '')
    description = data.get('description', '')
    lang = data.get('lang', 'de')
    codelist_values = data.get('codelist_values', None)
    
    # Generate AI-powered description
    try:
        generated_description = generate_concept_description(
            column=column,
            lang=lang,
            title=title,
            description=description,
            codelist_values=codelist_values,
            use_openai=True
        )
        
        return jsonify({'description': generated_description})
    
    except Exception as e:
        print(f"Error generating description: {e}")
        # Fallback to default description
        fallback_description = generate_default_description(column, lang)
        return jsonify({'description': fallback_description, 'fallback': True})

@app.route('/api/submit-concept', methods=['POST'])
def submit_concept():
    """Submit a concept to the I14Y API"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired'}), 404
    
    # Get the concept data from the request
    data = request.get_json() or {}
    
    # Extract required data
    index = data.get('index')
    if index is None or index < 0 or index >= len(session_data.get('columns', [])):
        return jsonify({'error': 'Invalid column index'}), 400
    
    column = session_data['columns'][index]
    
    # Get agency and contact info from request (provided by user at submission time)
    agency = data.get('agency', '')
    responsible_person = data.get('responsible_person', '')
    responsible_deputy = data.get('responsible_deputy', '')
    
    if not agency or not responsible_person or not responsible_deputy:
        return jsonify({'error': 'Agency and contact information are required'}), 400
    
    # Build form_data with submission-time information
    form_data = {
        'publisher_id': agency,
        'responsible_person': responsible_person,
        'responsible_deputy': responsible_deputy,
        'theme': session_data['form_data'].get('theme', '')
    }
    
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.replace('Bearer ', '').strip() if auth_header.startswith('Bearer ') else auth_header.strip()
    
    if not token:
        return jsonify({'error': 'Authorization token is required'}), 401
    
    # Extract concept parameters
    title = data.get('title', '')
    description = data.get('description', '')
    translations = data.get('translations', {})
    keywords = data.get('keywords', '')
    custom_identifier = data.get('identifier', '')
    
    # Extract type-specific parameters
    params = {}
    if column['type'] == 'Number':
        params['min_value'] = data.get('min_value', 0)
        params['max_value'] = data.get('max_value', 999)
        params['decimals'] = data.get('decimals', 0)
        params['unit'] = data.get('unit', '')
    elif column['type'] == 'String':
        params['max_length'] = data.get('max_length', 255)
        params['min_length'] = data.get('min_length', 0)
        params['pattern'] = data.get('pattern', '')
    elif column['type'] == 'Date':
        params['pattern'] = data.get('pattern', 'YYYY-MM-DD')
    
    # Generate the concept JSON
    concept_json = generate_concept_json(
        column=column,
        form_data=form_data,
        description=description,
        params=params,
        translations=translations,
        keywords=keywords,
        custom_identifier=custom_identifier
    )
       
    # Prepare the API request
    url = 'https://api.i14y.admin.ch/api/partner/v1/concepts'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.post(url, headers=headers, json=concept_json, timeout=(5, 30))
        
        # Handle different response codes
        if response.status_code == 201:  # Created
            # Extract location header for concept GUID
            location = response.headers.get('Location', '')
            # Extract GUID from location (e.g., "/api/partner/v1/concepts/abc-123-def")
            guid = location.split('/')[-1] if location else ''
            
            # Mark this concept as submitted in session data with GUID
            if 'submitted_concepts' not in session_data:
                session_data['submitted_concepts'] = {}
            session_data['submitted_concepts'][index] = guid
            try:
                save_session_data(session_data, session['session_data_id'])
            except Exception as e:
                print(f"Warning: Could not save session data after submission: {e}")
                # Don't fail the request if session save fails - concept was already created
            
            return jsonify({
                'success': True,
                'message': 'Concept created successfully',
                'location': location,
                'guid': guid
            })
        
        elif response.status_code == 409:  # Conflict - concept already exists
            try:
                conflict_data = response.json()
                return jsonify({
                    'success': True,
                    'message': 'Concept already exists',
                    'conflict': True,
                    'existing_concept': conflict_data
                })
            except:
                return jsonify({
                    'success': False,
                    'message': 'Concept already exists but could not parse conflict details'
                }), 409
        
        else:
            # Handle other error codes
            error_message = "Unknown error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_message = error_data.get('message', error_data.get('error', response.text))
                else:
                    error_message = str(error_data)
            except:
                error_message = response.text
            
            return jsonify({
                'success': False,
                'message': f"API error: {error_message}",
                'status': response.status_code
            }), response.status_code
    
    except requests.RequestException as e:
        logging.exception(f"Network error: {e}")
        return jsonify({
            'success': False,
            'message': 'Network error occurred. Please try again or contact support.'
        }), 500
    
    except Exception as e:
        logging.exception(f"Error in import process: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal error occurred. Please try again or contact support.'
        }), 500

@app.route('/results', methods=['GET'])
def results():
    if 'session_data_id' not in session:
        return redirect(url_for('upload_file'))
    
    # Get session data from file
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session data expired, corrupted, or not found - clean up the session file
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        # Flash error message and redirect
        return redirect(url_for('upload_file'))
    
    return render_template('results.html', 
                          columns=session_data['columns'],
                          form_data=session_data['form_data'],
                          themes=VALID_THEMES,  # Pass themes to the template
                          submitted_concepts=session_data.get('submitted_concepts', {}))

@app.route('/concept/<int:index>', methods=['GET'])
def view_concept(index):
    if 'session_data_id' not in session:
        return redirect(url_for('upload_file'))
    
    # Get session data from file
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session data expired or not found - clean up the session file
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return redirect(url_for('upload_file'))
    
    # Validate index
    if index < 0 or index >= len(session_data['columns']):
        return redirect(url_for('results'))
    
    # Get saved concept data if it exists
    concept_data = session_data.get('concept_data', {}).get(str(index), {})
    
    return render_template('concept.html', 
                          column=session_data['columns'][index],
                          index=index,
                          total=len(session_data['columns']),
                          form_data=session_data['form_data'],
                          saved_data=concept_data)  # Pass saved concept data to template

@app.route('/api/save-concept-data/<int:index>', methods=['POST'])
def save_concept_data(index):
    """Save concept form data for a specific column"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired'}), 404
    
    if index < 0 or index >= len(session_data.get('columns', [])):
        return jsonify({'error': 'Invalid column index'}), 400
    
    # Get the data from the request - handle both JSON and other content types
    try:
        data = request.get_json() or {}
    except Exception:
        # If JSON parsing fails, try to get form data
        data = request.form.to_dict() or {}
    
    # Initialize concept_data dict if it doesn't exist
    if 'concept_data' not in session_data:
        session_data['concept_data'] = {}
    
    # Save the data for this specific concept
    session_data['concept_data'][str(index)] = data
    
    # Save back to file with error handling
    try:
        save_session_data(session_data, session_id=session['session_data_id'])
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving concept data: {e}")
        return jsonify({'error': 'Failed to save concept data'}), 500

def generate_concept_json(column, form_data, description, params=None, translations=None, dataset_metadata=None, keywords=None, custom_identifier=None):
    """Generate a concept JSON based on column information and user inputs"""
    # Default params if none provided
    if not params:
        params = {}
    
    # Handle keywords - prioritize multilingual translations over simple keyword string
    if translations and translations.get('keywords'):
        # Use keywords from translations if available
        # translations['keywords'] is an object like: {de: "kw1, kw2", en: "kw1, kw2", ...}
        concept_keywords = []
        keywords_trans = translations['keywords']
        
        print(f"DEBUG - Processing keywords from translations: {keywords_trans}")
        
        # Parse keywords per language
        keywords_by_lang = {}
        for lang in ['de', 'en', 'fr', 'it', 'rm']:
            if keywords_trans.get(lang) and keywords_trans[lang].strip():
                # Split by comma to get individual keywords
                keywords_by_lang[lang] = [kw.strip() for kw in keywords_trans[lang].split(',') if kw.strip()]
            else:
                keywords_by_lang[lang] = []
        
        print(f"DEBUG - Parsed keywords by language: {keywords_by_lang}")
        
        # Find the maximum number of keywords across all languages
        max_keywords = max(len(keywords_by_lang[lang]) for lang in keywords_by_lang)
        
        # Create multilingual keyword objects with proper structure (label + uri)
        for i in range(max_keywords):
            keyword_label = {}
            for lang in ['de', 'en', 'fr', 'it', 'rm']:
                if i < len(keywords_by_lang[lang]):
                    keyword_label[lang] = keywords_by_lang[lang][i]
                else:
                    # If a language has fewer keywords, use the German one as fallback
                    keyword_label[lang] = keywords_by_lang['de'][i] if i < len(keywords_by_lang['de']) else ''
            
            # Only add if at least one language has content
            if any(keyword_label.values()):
                concept_keywords.append({
                    "label": keyword_label,
                    "uri": None
                })
        
        print(f"DEBUG - Final concept_keywords: {concept_keywords}")
    elif keywords and keywords.strip():
        # Parse keywords string into multilingual objects (fallback)
        keyword_list = []
        for kw in keywords.split(','):
            kw = kw.strip()
            if kw:
                keyword_list.append({
                    "label": simple_multilingual(kw),
                    "uri": None
                })
        concept_keywords = keyword_list
    else:
        # Generate keywords from column name parts (fallback)
        concept_keywords = []
        for word in column['name'].split('_'):
            if word:
                # Only use simple keywords without translations by default
                concept_keywords.append({
                    "label": simple_multilingual(word),
                    "uri": None
                })
    
    # Ensure we have at least one keyword
    if not concept_keywords:
        concept_keywords = [{
            "label": simple_multilingual(column['name']),
            "uri": None
        }]
    
    # Determine concept type - using correct I14Y API enum values
    concept_type = "String"  # Default - changed from "Text" to "String"
    if column['is_codelist']:
        concept_type = "CodeList"
    elif column['type'] == "Number":
        concept_type = "Numeric"
    elif column['type'] == "Date":
        concept_type = "Date"  # Changed from "DateTime" to "Date"
    
    # Create identifier with proper format - use custom identifier if provided
    if custom_identifier and custom_identifier.strip():
        identifier = custom_identifier.strip().upper()
    else:
        identifier = column['name'].replace(' ', '_').upper()
    
    # Build data object with exact field ordering
    data = {"conceptType": concept_type}
    
    # Add type-specific fields
    if column['is_codelist']:
        # Use "String" instead of "Text" - matching the exact enum values from the API
        value_type = "Numeric" if column['code_type'] == 'numeric' else "String"
        data["codeListEntryDefaultSortProperty"] = "Code"
        data["codeListEntryValueType"] = value_type
        data["codeListEntryValueMaxLength"] = column.get('max_length', 0)
    
    elif column['type'] == "Number":
        data["maxValue"] = int(params.get('max_value', 999))
        data["measurementUnit"] = params.get('unit', '')
        data["minValue"] = int(params.get('min_value', 0))
        data["numberDecimals"] = int(params.get('decimals', 0))
        # Note: pattern is not supported for Numeric type in I14Y API
    
    elif column['type'] == "Date":
        # Date type only has pattern field (not dateTimeFormat)
        pattern = params.get('pattern') or params.get('format') or column.get('pattern') or column.get('format', 'yyyy-mm-dd')
        data["pattern"] = pattern
    
    else:  # String type
        # String type requires maxLength and minLength
        data["maxLength"] = int(params.get('max_length', 255))
        data["minLength"] = int(params.get('min_length', 0))
        # Include pattern if provided and not empty (allow .* as valid pattern)
        pattern = params.get('pattern')
        if pattern and pattern.strip():
            data["pattern"] = pattern.strip()
    
    # Empty conformsTo field instead of adding a standard entry
    data["conformsTo"] = []
    
    # Use translations if available, otherwise use simple multilingual text
    if translations and translations.get('description'):
        data["description"] = translations['description']
    else:
        data["description"] = simple_multilingual(description)
    
    data["identifier"] = identifier
    data["keywords"] = concept_keywords
    
    # Use translations for name if available
    if translations and translations.get('title'):
        data["name"] = translations['title']
    else:
        data["name"] = simple_multilingual(column['name'])
    
    data["publisher"] = {
        "identifier": form_data['publisher_id']
    }
    
    # Ensure email format for responsible persons - API expects email, not identifier
    data["responsibleDeputy"] = {
        "email": form_data['responsible_deputy']
    }
    data["responsiblePerson"] = {
        "email": form_data['responsible_person']
    }
    
    # Ensure themes is correctly formatted
    data["themes"] = [{
        "code": form_data['theme']
    }]
    
    # Format dates properly
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
    
    data["validFrom"] = now
    data["validTo"] = "9999-12-31T23:59:59.9999999+00:00"
    data["version"] = "1.0.0"
    
    return {"data": data}

def generate_concept_description(column, lang="de", title="", description="", codelist_values=None, use_openai=False):
    """Generate a description using OpenAI API or fallback to default description"""
    if use_openai:
        try:
            # Get OpenAI API key from environment
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                # Fallback to default generation if no API key
                return generate_default_description(column, lang)
            
            # Set up OpenAI client
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Build context for OpenAI
            context = f"Column name: {column['name']}\n"
            context += f"Data type: {column['type']}\n"
            context += f"Is codelist: {column['is_codelist']}\n"
            
            if title:
                context += f"Title: {title}\n"
            if description:
                context += f"Existing description: {description}\n"
            
            if column['is_codelist'] and codelist_values:
                context += f"Codelist values: {', '.join(str(v) for v in codelist_values[:10])}\n"
                if len(codelist_values) > 10:
                    context += f"... and {len(codelist_values) - 10} more values\n"
            
            if column.get('pattern'):
                context += f"Pattern: {column['pattern']}\n"
            
            # Create prompt based on language
            lang_prompts = {
                "de": "Generiere eine präzise, professionelle Beschreibung für dieses Datenkonzept auf Deutsch. Die Beschreibung sollte klar und informativ sein.",
                "en": "Generate a precise, professional description for this data concept in English. The description should be clear and informative.",
                "fr": "Générez une description précise et professionnelle pour ce concept de données en français. La description doit être claire et informative.",
                "it": "Genera una descrizione precisa e professionale per questo concetto di dati in italiano. La descrizione dovrebbe essere chiara e informativa."
            }
            
            prompt = lang_prompts.get(lang.lower(), lang_prompts["de"])
            prompt += "\n\nKontext:\n" + context
            prompt += "\n\nBeschreibung:"
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates precise descriptions for data concepts in statistical and administrative contexts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            generated_description = response.choices[0].message.content.strip()
            
            # Clean up the response
            if generated_description.startswith('"') and generated_description.endswith('"'):
                generated_description = generated_description[1:-1]
            
            return generated_description
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to default generation
            return generate_default_description(column, lang)
    else:
        # Use default description generation
        return generate_default_description(column, lang)

def generate_default_description(column, lang="de"):
    """Generate a default description based on column analysis"""
    if column['is_codelist']:
        if lang == "fr":
            description = f"Le concept {column['name']} contient une liste de codes"
            if column.get('unique_values') and len(column['unique_values']) > 0:
                examples = ', '.join(str(v) for v in column['unique_values'][:3])
                description += f". Les valeurs incluent: {examples}"
                if len(column['unique_values']) > 3:
                    description += "..."
        elif lang == "it":
            description = f"Il concetto {column['name']} contiene un elenco di codici"
            if column.get('unique_values') and len(column['unique_values']) > 0:
                examples = ', '.join(str(v) for v in column['unique_values'][:3])
                description += f". I valori includono: {examples}"
                if len(column['unique_values']) > 3:
                    description += "..."
        elif lang == "en":
            description = f"The concept {column['name']} contains a codelist"
            if column.get('unique_values') and len(column['unique_values']) > 0:
                examples = ', '.join(str(v) for v in column['unique_values'][:3])
                description += f". Values include: {examples}"
                if len(column['unique_values']) > 3:
                    description += "..."
        else:  # Default to German
            description = f"Das Konzept {column['name']} beinhaltet eine Codeliste"
            if column.get('unique_values') and len(column['unique_values']) > 0:
                examples = ', '.join(str(v) for v in column['unique_values'][:3])
                description += f". Darin sind folgende Werte verzeichnet: {examples}"
                if len(column['unique_values']) > 3:
                    description += "..."
    elif column['type'] == 'Date':
        if lang == "fr":
            description = f"Le concept {column['name']} contient des informations de date"
            if column.get('format'):
                description += f" au format {column['format']}"
        elif lang == "it":
            description = f"Il concetto {column['name']} contiene informazioni sulla data"
            if column.get('format'):
                description += f" nel formato {column['format']}"
        elif lang == "en":
            description = f"The concept {column['name']} contains date information"
            if column.get('format'):
                description += f" in format {column['format']}"
        else:  # Default to German
            description = f"Das Konzept {column['name']} enthält Datumsinformationen"
            if column.get('format'):
                description += f" im Format {column['format']}"
    elif column['type'] == 'Number':
        if lang == "fr":
            description = f"Le concept {column['name']} contient des valeurs numériques"
        elif lang == "it":
            description = f"Il concetto {column['name']} contiene valori numerici"
        elif lang == "en":
            description = f"The concept {column['name']} contains numeric values"
        else:  # Default to German
            description = f"Das Konzept {column['name']} enthält numerische Werte"
    else:
        if lang == "fr":
            description = f"Le concept {column['name']} contient des données textuelles"
        elif lang == "it":
            description = f"Il concetto {column['name']} contiene dati testuali"
        elif lang == "en":
            description = f"The concept {column['name']} contains text data"
        else:  # Default to German
            description = f"Das Konzept {column['name']} enthält Textdaten"
    
    if column.get('pattern'):
        if lang == "fr":
            description += f". Modèle: {column['pattern']}"
        elif lang == "it":
            description += f". Pattern: {column['pattern']}"
        elif lang == "en":
            description += f". Pattern: {column['pattern']}"
        else:  # Default to German
            description += f". Muster: {column['pattern']}"
    
    return description

@app.route('/api/extract-keywords/<int:index>', methods=['POST'])
def extract_keywords(index):
    """Extract keywords from title and description using OpenAI API"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No results found. Please upload a file first.'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    
    if index < 0 or index >= len(session_data['columns']):
        return jsonify({'error': 'Invalid column index'}), 400
    
    column = session_data['columns'][index]
    
    # Get data from request
    data = request.get_json() or {}
    
    # Extract parameters
    title = data.get('title', '')
    description = data.get('description', '')
    lang = data.get('lang', 'de').lower()
    
    # Validate language is supported
    if lang not in ['de', 'en', 'fr', 'it']:
        lang = 'de'  # Default to German if unsupported language
    
    # Generate keywords using OpenAI
    try:
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not configured'}), 500
        
        # Set up OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Build context for OpenAI
        context = f"Column name: {column['name']}\n"
        context += f"Data type: {column['type']}\n"
        context += f"Is codelist: {column['is_codelist']}\n"
        
        if title:
            context += f"Title: {title}\n"
        if description:
            context += f"Description: {description}\n"
        
        # Create prompt based on language
        lang_prompts = {
            "de": "Extrahiere 3-7 relevante Keywords aus dem folgenden Kontext. Die Keywords sollten durch Kommas getrennt sein und für die statistische Datenbeschreibung relevant sein.",
            "en": "Extract 3-7 relevant keywords from the following context. The keywords should be separated by commas and relevant for statistical data description.",
            "fr": "Extrayez 3-7 mots-clés pertinents du contexte suivant. Les mots-clés doivent être séparés par des virgules et pertinents pour la description des données statistiques.",
            "it": "Estrai 3-7 parole chiave rilevanti dal seguente contesto. Le parole chiave dovrebbero essere separate da virgole e rilevanti per la descrizione dei dati statistici."
        }
        
        prompt = lang_prompts.get(lang, lang_prompts["de"])
        prompt += "\n\nKontext:\n" + context
        prompt += "\n\nKeywords:"
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts relevant keywords for statistical data concepts. Return only the keywords separated by commas, no additional text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        keywords_text = response.choices[0].message.content.strip()
        
        # Clean up the response - remove quotes if present
        if keywords_text.startswith('"') and keywords_text.endswith('"'):
            keywords_text = keywords_text[1:-1]
        elif keywords_text.startswith("'") and keywords_text.endswith("'"):
            keywords_text = keywords_text[1:-1]
        
        return jsonify({
            'keywords': keywords_text
        })
        
    except Exception as e:
        print(f"OpenAI keyword extraction error: {e}")
        # Fallback to simple keyword extraction
        return jsonify({
            'keywords': generate_fallback_keywords(title, description, lang)
        })

def generate_fallback_keywords(title, description, lang):
    """Generate fallback keywords when OpenAI is not available"""
    # Simple keyword extraction based on title and description
    text = f"{title} {description}".strip()
    
    if not text:
        return ""
    
    # Split into words and filter
    words = text.split()
    
    # Remove common stop words based on language
    stop_words = {
        'de': ['der', 'die', 'das', 'und', 'oder', 'mit', 'für', 'von', 'zu', 'im', 'am', 'auf', 'in', 'an', 'bei', 'nach', 'aus', 'über', 'unter', 'vor', 'hinter', 'neben', 'zwischen', 'durch', 'gegen', 'um', 'ohne', 'seit', 'bis', 'während', 'wegen', 'statt', 'trotz', 'obwohl', 'obgleich', 'wenn', 'falls', 'sofern', 'soweit', 'sobald', 'bevor', 'nachdem', 'seitdem', 'solange', 'während', 'als', 'wie', 'da', 'weil', 'denn', 'deshalb', 'darum', 'deswegen', 'also', 'folglich', 'somit', 'demnach', 'mithin', 'sondern', 'doch', 'jedoch', 'allerdings', 'hingegen', 'andererseits', 'im gegenteil', 'vielmehr', 'zwar', 'zwar...aber', 'entweder...oder', 'weder...noch', 'nicht nur...sondern auch', 'sowohl...als auch', 'teils...teils', 'halb...halb', 'bald...bald', 'mal...mal', 'einmal...einmal', 'bald...bald', 'teils...teils', 'halb...halb', 'bald...bald', 'mal...mal', 'einmal...einmal'],
        'en': ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how'],
        'fr': ['le', 'la', 'les', 'l\'', 'un', 'une', 'des', 'du', 'de', 'à', 'au', 'aux', 'en', 'dans', 'sur', 'sous', 'avec', 'sans', 'pour', 'par', 'chez', 'contre', 'entre', 'vers', 'depuis', 'pendant', 'avant', 'après', 'jusque', 'lorsque', 'quand', 'comme', 'si', 'que', 'qui', 'quoi', 'dont', 'où', 'd\'où', 'parce que', 'puisque', 'car', 'donc', 'or', 'ni', 'mais', 'et', 'ou', 'ne', 'pas', 'plus', 'très', 'trop', 'peu', 'beaucoup', 'assez', 'autant', 'tant', 'tellement', 'si', 'tel', 'telle', 'tels', 'telles', 'tout', 'toute', 'tous', 'toutes', 'chaque', 'chacun', 'chacune', 'aucun', 'aucune', 'nul', 'nulle', 'rien', 'personne', 'rien', 'autre', 'autres', 'même', 'mêmes', 'quel', 'quelle', 'quels', 'quelles', 'ce', 'cet', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs'],
        'it': ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'del', 'dello', 'della', 'dei', 'degli', 'delle', 'al', 'allo', 'alla', 'ai', 'agli', 'alle', 'dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle', 'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'e', 'o', 'ma', 'se', 'che', 'chi', 'cosa', 'come', 'quando', 'dove', 'perché', 'perchè', 'quindi', 'allora', 'poi', 'dopo', 'prima', 'mentre', 'finché', 'finchè', 'appena', 'non', 'anche', 'solo', 'soltanto', 'pure', 'neppure', 'nemmeno', 'neanche', 'mai', 'sempre', 'spesso', 'raramente', 'talvolta', 'qualche', 'alcuni', 'alcune', 'tutti', 'tutte', 'ogni', 'ciascun', 'ciascuna', 'nessun', 'nessuna', 'nulla', 'nessuno', 'altro', 'altra', 'altri', 'altre', 'stesso', 'stessa', 'stessi', 'stesse', 'tale', 'tali', 'qual', 'quale', 'quali', 'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle', 'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue', 'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'nostri', 'nostre', 'vostro', 'vostra', 'vostri', 'vostre', 'loro']
    }
    
    stop_words_list = stop_words.get(lang, stop_words['de'])
    
    # Filter words: remove stop words, keep only alphanumeric words, limit length
    keywords = []
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        if (len(word_lower) > 2 and 
            word_lower not in stop_words_list and 
            word_lower.isalnum()):
            keywords.append(word_lower)
    
    # Remove duplicates and limit to 5 keywords
    unique_keywords = list(dict.fromkeys(keywords))[:5]
    
    return ', '.join(unique_keywords)

@app.route('/api/dataset/columns', methods=['GET'])
def list_dataset_columns():
    """Return analyzed columns so the UI can offer manual selection."""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found. Please upload a file first.'}), 404
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    return jsonify({
        'success': True,
        'columns': summarize_columns_for_response(session_data.get('columns', []))
    })

@app.route('/api/codelist/label-columns/<int:index>', methods=['GET'])
def list_label_column_candidates(index):
    """Suggest label columns (with guessed languages) for a codelist."""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found. Please upload a file first.'}), 404
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    columns = session_data.get('columns', [])
    if index < 0 or index >= len(columns):
        return jsonify({'error': 'Invalid column index'}), 400
    target_column = columns[index]
    if not target_column.get('is_codelist'):
        return jsonify({'error': 'Selected column is not a codelist'}), 400
    df = load_uploaded_dataframe(session_data)
    if df is None or target_column['name'] not in df.columns:
        return jsonify({'error': 'Unable to load dataset for the current session'}), 400

    candidates = []
    for column_name in df.columns:
        if column_name == target_column['name']:
            continue
        lang = infer_language_from_column(column_name) or 'unknown'
        sample_values = []
        for value in df[column_name].dropna().astype(str).head(3):
            value = value.strip()
            if value:
                sample_values.append(value)
        candidates.append({
            'name': column_name,
            'language': lang,
            'samples': sample_values
        })

    return jsonify({
        'success': True,
        'codeColumn': target_column['name'],
        'candidates': candidates
    })

@app.route('/api/codelist/add-labels', methods=['POST'])
def add_codelist_labels():
    """Combine code values with multilingual labels sourced from other columns."""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found. Please upload a file first.'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    
    data = request.get_json() or {}
    code_column = data.get('codeColumn')
    label_columns = data.get('labelColumns', [])
    
    if not code_column or not label_columns:
        return jsonify({'error': 'Both codeColumn and labelColumns are required'}), 400
    
    df = load_uploaded_dataframe(session_data)
    if df is None or code_column not in df.columns:
        return jsonify({'error': 'Unable to access the requested columns in the uploaded file'}), 400
    
    target_meta = next((col for col in session_data.get('columns', []) if col.get('name') == code_column), None)
    if not target_meta or not target_meta.get('is_codelist'):
        return jsonify({'error': f'Column "{code_column}" is not registered as a codelist'}), 400
    
    parsed_columns = []
    for entry in label_columns:
        if isinstance(entry, dict):
            col_name = entry.get('name')
            lang = entry.get('lang')
        else:
            col_name = entry
            lang = None
        if not col_name or col_name not in df.columns:
            return jsonify({'error': f'Column "{col_name}" is not available in the dataset'}), 400
        lang = (lang or infer_language_from_column(col_name) or 'de').lower()
        parsed_columns.append({'name': col_name, 'lang': lang})
    
    collected = {}
    languages = set()
    required_columns = [code_column] + [c['name'] for c in parsed_columns]
    for _, row in df[required_columns].iterrows():
        code_value = row[code_column]
        if pd.isna(code_value):
            continue
        code_key = str(code_value).strip()
        if not code_key:
            continue
        entry = collected.setdefault(code_key, {})
        for column in parsed_columns:
            value = row[column['name']]
            if pd.isna(value):
                continue
            label = str(value).strip()
            if label:
                entry[column['lang']] = label
                languages.add(column['lang'])
    
    labels = [{'code': code, 'labels': lang_map} for code, lang_map in collected.items() if lang_map]
    
    return jsonify({
        'success': True,
        'codeColumn': code_column,
        'languages': sorted(languages),
        'labels': labels
    })

def summarize_columns_for_response(columns):
    """Return lightweight metadata for the frontend."""
    summary = []
    for col in columns:
        summary.append({
            'name': col.get('name'),
            'type': col.get('type'),
            'isCodelist': col.get('is_codelist', False),
            'guessedLanguage': infer_language_from_column(col.get('name', ''))
        })
    return summary

@app.route('/api/add-codelist-entries/<concept_guid>', methods=['POST'])
def add_codelist_entries(concept_guid):
    """Add codelist entries to an existing concept"""
    # Get the payload from the request
    payload = request.json
    
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid Authorization header'}), 400
    
    if not concept_guid:
        return jsonify({'error': 'Missing concept GUID'}), 400
    
    if not payload or 'data' not in payload or not isinstance(payload['data'], list):
        return jsonify({'error': 'Invalid payload format'}), 400
    
    # Translate monolingual labels if needed
    try:
        from utils.translator import create_multilingual_text
        
        # Check if we need to perform translations
        needs_translation = False
        for entry in payload['data']:
            # If 'name' exists and is a string (monolingual) instead of an object (multilingual)
            if 'name' in entry and isinstance(entry['name'], str):
                needs_translation = True
                break
                
        if needs_translation:
            translated_entries = []
            
            for entry in payload['data']:
                translated_entry = entry.copy()  # Create a copy to modify
                
                # Translate the name if it's a string
                if 'name' in entry and isinstance(entry['name'], str):
                    label = entry['name']
                    
                    try:
                        # Detect the source language instead of assuming DE
                        source_lang = safe_detect_language(label, default='de')
                        
                        # Create translations for all required languages
                        translated_labels = create_multilingual_text(label, source_lang)
                        translated_entry['name'] = translated_labels
                    except Exception:
                        # Fall back to simple multilingual with the same text for all languages
                        translated_entry['name'] = simple_multilingual(label)
                
                translated_entries.append(translated_entry)
            
            # Replace the original entries with translated ones
            payload['data'] = translated_entries
    except ImportError:
        pass
    except Exception as e:
        logging.exception(f"Error in concept import setup: {e}")
    
    # Make the request to the I14Y API
    url = f'https://api.i14y.admin.ch/api/partner/v1/concepts/{concept_guid}/codelist-entries/imports/json'
    headers = {
        'accept': '*/*',
        'Authorization': auth_header
    }
    
    try:
        import tempfile
        import json
        # Create a temporary file with the JSON payload
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False)
            temp_path = temp_file.name
        
        # Open the file in binary mode for the request
        with open(temp_path, 'rb') as f:
            files = {'file': ('payload.json', f, 'application/json')}
            # Make the actual API call
            # URL host/scheme are hardcoded to api.i14y.admin.ch (allowlisted); only
            # the path segment {concept_guid} varies. Not user-controlled SSRF.
            response = requests.post(url, headers=headers, files=files, timeout=(10, 60))  # nosemgrep: python.flask.security.injection.ssrf-requests.ssrf-requests
        
        # Delete the temporary file
        import os
        os.unlink(temp_path)
        
        # Check for errors
        if response.status_code >= 400:
            error_message = "Unknown error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_message = error_data.get('message', error_data.get('error', response.text))
                else:
                    error_message = str(error_data)
            except:
                error_message = response.text
            
            return jsonify({
                'success': False,
                'message': f"API error: {error_message}",
                'status': response.status_code
            }), response.status_code
        
        # Return success
        translated = "with automatic translation" if needs_translation else ""
        return jsonify({
            'success': True,
            'message': f"Successfully added {len(payload['data'])} codelist entries {translated}",
            'status': response.status_code
        })
        
    except Exception as e:
        logging.exception(f"Error publishing draft: {e}")
        return jsonify({'error': 'Failed to publish draft. Please try again or contact support.'}), 500

@app.before_request
def cleanup_old_sessions():
    """Clean up session files older than 1 hour"""
    import random
    if random.random() > 0.01:
        return
    
    import time
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    for filename in os.listdir(SESSION_DATA_DIR):
        if filename.endswith('.pkl'):
            file_path = os.path.join(SESSION_DATA_DIR, filename)
            if os.path.getmtime(file_path) < one_hour_ago:
                try:
                    # Reuse the signed loader so we never pickle.load() an
                    # untrusted / tampered file even during cleanup.
                    session_id = filename[:-4]  # strip '.pkl'
                    stored_data = load_session_data(session_id)
                    if stored_data:
                        dataset_info = stored_data.get('dataset_file', {})
                        dataset_path = dataset_info.get('path') if isinstance(dataset_info, dict) else dataset_info
                        if dataset_path and os.path.exists(dataset_path):
                            os.remove(dataset_path)
                except Exception:
                    pass
                try:
                    os.remove(file_path)
                except Exception:
                    pass

# Add this function to generate codes
def generate_code_for_value(value, code_length=2, existing_codes=None, known_mappings=None):
    """Generate a meaningful code for a given value."""
    if not existing_codes:
        existing_codes = set()
    
    if not known_mappings:
        known_mappings = {}
    
    # If we already have a known mapping for this value, use it
    if value in known_mappings:
        code = known_mappings[value]
        if len(code) > code_length:
            code = code[:code_length]
        
        if code not in existing_codes:
            return code
    
    # Normalize the value: convert to uppercase, remove accents, remove special chars
    # This helps with non-ASCII characters like ü, é, etc.
    normalized = unicodedata.normalize('NFKD', str(value))
    normalized = ''.join([c for c in normalized if not unicodedata.combining(c)])
    normalized = normalized.upper()
    normalized = ''.join([c for c in normalized if c.isalnum() or c.isspace()])
    
    # Method 1: For short values, use the whole value if it fits
    if len(normalized) <= code_length and not normalized.isdigit():
        code = normalized
        if code not in existing_codes:
            return code
    
    # Method 2: For multi-word values, use initials
    words = normalized.split()
    if len(words) >= 2:
        initials = ''.join([word[0] for word in words])
        if len(initials) <= code_length:
            if initials not in existing_codes:
                return initials
            
            # Try to use first 2 letters from first word and first letter from others
            if len(words[0]) >= 2:
                extended_initials = words[0][:2] + ''.join([word[0] for word in words[1:]])
                if len(extended_initials) <= code_length:
                    if extended_initials not in existing_codes:
                        return extended_initials
    
    # Method 3: Take first n letters
    if len(normalized) > code_length:
        prefix = normalized.replace(' ', '')[:code_length]
        if prefix not in existing_codes:
            return prefix
    
    # Method 4: For numeric values, pad with zeros
    if normalized.isdigit():
        numeric_code = normalized.zfill(code_length)[:code_length]
        if numeric_code not in existing_codes:
            return numeric_code
    
    # Method 5: Add numbers to disambiguate
    base_code = normalized.replace(' ', '')[:code_length-1] if len(normalized) >= code_length else normalized.replace(' ', '')
    for i in range(1, 10):  # Try adding 1-9 to the end
        suffix_code = (base_code + str(i))[:code_length]
        if suffix_code not in existing_codes:
            return suffix_code
    
    # Last resort: generate a random code
    import random
    letters = string.ascii_uppercase
    while True:
        random_code = ''.join(random.choice(letters) for _ in range(code_length))
        if random_code not in existing_codes:
            return random_code

def get_known_code_mappings():
    """Return known code mappings for common entities."""
    # Country codes (ISO 3166-1 alpha-2)
    countries = {
        "Schweiz": "CH", "Switzerland": "CH", "Suisse": "CH", "Svizzera": "CH",
        "Deutschland": "DE", "Germany": "DE", "Allemagne": "DE", "Germania": "DE",
        "Österreich": "AT", "Austria": "AT", "Autriche": "AT", "Austria": "AT",
        "Frankreich": "FR", "France": "FR", "Francia": "FR",
        "Italien": "IT", "Italy": "IT", "Italie": "IT", "Italia": "IT",
        "Vereinigte Staaten": "US", "United States": "US", "États-Unis": "US", "Stati Uniti": "US",
        "Vereinigtes Königreich": "GB", "United Kingdom": "GB", "Royaume-Uni": "GB", "Regno Unito": "GB",
        "Spanien": "ES", "Spain": "ES", "Espagne": "ES", "Spagna": "ES",
        "Niederlande": "NL", "Netherlands": "NL", "Pays-Bas": "NL", "Paesi Bassi": "NL",
        "Belgien": "BE", "Belgium": "BE", "Belgique": "BE", "Belgio": "BE",
        "Schweden": "SE", "Sweden": "SE", "Suède": "SE", "Svezia": "SE",
        "Norwegen": "NO", "Norway": "NO", "Norvège": "NO", "Norvegia": "NO",
        "Dänemark": "DK", "Denmark": "DK", "Danemark": "DK", "Danimarca": "DK",
        "Finnland": "FI", "Finland": "FI",
        "Portugal": "PT", "Portogallo": "PT",
        "Griechenland": "GR", "Greece": "GR", "Grèce": "GR", "Grecia": "GR",
        "Polen": "PL", "Poland": "PL", "Pologne": "PL",
        "Tschechien": "CZ", "Czech Republic": "CZ", "République tchèque": "CZ", "Repubblica Ceca": "CZ",
        "Ungarn": "HU", "Hungary": "HU", "Hongrie": "HU", "Ungheria": "HU",
        "Russland": "RU", "Russia": "RU", "Russie": "RU", "Russia": "RU",
        "Brasilien": "BR", "Brazil": "BR", "Brésil": "BR", "Brasile": "BR",
        "Australien": "AU", "Australia": "AU", "Australie": "AU", "Australia": "AU",
        "Kanada": "CA", "Canada": "CA", "Canada": "CA", "Canada": "CA",
    }
    
    # Language codes (ISO 639-1)
    languages = {
        "Deutsch": "DE", "German": "DE", "Allemand": "DE", "Tedesco": "DE",
        "Englisch": "EN", "English": "EN", "Anglais": "EN", "Inglese": "EN",
        "Französisch": "FR", "French": "FR", "Français": "FR", "Francese": "FR",
        "Italienisch": "IT", "Italian": "IT", "Italien": "IT", "Italiano": "IT",
        "Spanisch": "ES", "Spanish": "ES", "Espagnol": "ES", "Spagnolo": "ES",
        "Portugiesisch": "PT", "Portuguese": "PT", "Portugais": "PT", "Portoghese": "PT",
        "Russisch": "RU", "Russian": "RU", "Russe": "RU", "Russo": "RU",
        "Chinesisch": "ZH", "Chinese": "ZH", "Chinois": "ZH", "Cinese": "ZH",
        "Japanisch": "JA", "Japanese": "JA", "Japonais": "JA", "Giapponese": "JA",
        "Arabisch": "AR", "Arabic": "AR", "Arabe": "AR", "Arabo": "AR"
    }
    
    # Days of the week
    days = {
        "Montag": "MO", "Monday": "MO", "Lundi": "MO", "Lunedì": "MO",
        "Dienstag": "TU", "Tuesday": "TU", "Mardi": "TU", "Martedì": "TU",
        "Mittwoch": "WE", "Wednesday": "WE", "Mercredi": "WE", "Mercoledì": "WE",
        "Donnerstag": "TH", "Thursday": "TH", "Jeudi": "TH", "Giovedì": "TH",
        "Freitag": "FR", "Friday": "FR", "Vendredi": "FR", "Venerdì": "FR",
        "Samstag": "SA", "Saturday": "SA", "Samedi": "SA", "Sabato": "SA",
        "Sonntag": "SU", "Sunday": "SU", "Dimanche": "SU", "Domenica": "SU"
    }
    
    # Months
   
    months = {
        "Januar": "01", "January": "01", "Janvier": "01", "Gennaio": "01",
        "Februar": "02", "February": "02", "Février": "02", "Febbraio": "02",
        "März": "03", "March": "03", "Mars": "03", "Marzo": "03",
        "April": "04", "Avril": "04", "Aprile": "04",
        "Mai": "05", "May": "05", "Mag": "05", "Maggio": "05",
        "Juni": "06", "June": "06", "Juin": "06", "Giugno": "06",
        "Juli": "07", "July": "07", "Juillet": "07", "Luglio": "07",
        "August": "08", "Août": "08", "Agosto": "08",
        "September": "09", "Septembre": "09", "Settembre": "09",
        "Oktober": "10", "October": "10", "Octobre": "10", "Ottobre": "10",
        "November": "11", "Novembre": "11",
        "Dezember": "12", "December": "12", "Décembre": "12", "Dicembre": "12"
    }
    
    # Boolean values
    boolean = {
        "Ja": "Y", "Yes": "Y", "Oui": "Y", "Si": "Y",
        "Nein": "N", "No": "N", "Non": "N",
        "Wahr": "T", "True": "T", "Vrai": "T", "Vero": "T", 
        "Falsch": "F", "False": "F", "Faux": "F", "Falso": "F"
    }
    
    # Common units
    units = {
        "Meter": "M", "Metre": "M", "Mètre": "M", "Metro": "M",
        "Kilometer": "KM", "Kilometre": "KM", "Kilomètre": "KM", "Chilometro": "KM",
        "Zentimeter": "CM", "Centimeter": "CM", "Centimètre": "CM", "Centimetro": "CM",
        "Millimeter": "MM", "Millimetre": "MM", "Millimètre": "MM", "Millilitro": "MM",
        "Kilogramm": "KG", "Kilogram": "KG", "Kilogramme": "KG", "Chilogrammo": "KG",
        "Gramm": "G", "Gram": "G", "Gramme": "G", "Grammo": "G",
        "Liter": "L", "Litre": "L", "Litre": "L", "Litro": "L",
        "Milliliter": "ML", "Millilitre": "ML", "Millilitre": "ML", "Millilitro": "ML",
        "Stunde": "H", "Hour": "H", "Heure": "H", "Ora": "H",
        "Minute": "MIN", "Minute": "MIN", "Minuto": "MIN",
        "Sekunde": "SEC", "Second": "SEC", "Seconde": "SEC", "Secondo": "SEC",
        "Prozent": "%", "Percent": "%", "Pour cent": "%", "Percento": "%"
    }
    
    # Combine all mappings
    combined = {}
    combined.update(countries)
    combined.update(languages)
    combined.update(days)
    combined.update(months)
    combined.update(boolean)
    combined.update(units)
    
    return combined

def determine_value_type(values):
    """Try to determine what kind of values we're dealing with."""
    if not values:
        return "unknown"
    
    # Check if all values are countries
    known_mappings = get_known_code_mappings()
    countries = {k: v for k, v in known_mappings.items() if len(v) == 2 and v.isalpha() and v.isupper()}
    country_matches = sum(1 for v in values if v in countries)
    if country_matches > len(values) * 0.5:  # If more than half are countries
        return "countries"
    
    # Check if all values are languages
    language_match_prefixes = ["sprach", "language", "tongue", "idioma", "lingua"]
    if any(any(prefix in str(v).lower() for prefix in language_match_prefixes) for v in values):
        return "languages"
    
    # Check if we're dealing with days of the week
    days = ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    day_matches = sum(1 for v in values if str(v).lower() in days)
    if day_matches > 0 and day_matches == len(values):
        return "days"
    
    # Check if we're dealing with months
    months = ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember",
              "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    month_matches = sum(1 for v in values if str(v).lower() in months)
    if month_matches > 0 and month_matches == len(values):
        return "months"
    
    # Check if we're dealing with boolean values
    boolean_values = ["ja", "nein", "yes", "no", "true", "false", "wahr", "falsch"]
    boolean_matches = sum(1 for v in values if str(v).lower() in boolean_values)
    if boolean_matches > 0 and boolean_matches == len(values):
        return "boolean"
    
    # Check if numeric values
    if all(str(v).isdigit() for v in values):
        return "numeric"
    
    # Default to "text"
    return "text"

@app.route('/api/generate-codes/<int:index>', methods=['POST'])
def generate_codes(index):
    """Generate codes for codelist values"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found. Please upload a file first.'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        # Session expired - clean up and return error
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    
    if index < 0 or index >= len(session_data['columns']):
        return jsonify({'error': 'Invalid column index'}), 400
    
    column = session_data['columns'][index]
    
    if not column['is_codelist']:
        return jsonify({'error': 'This column is not a codelist'}), 400
    
    data = request.json
    code_length = data.get('codeLength', 2)
    
    if code_length < 1 or code_length > 10:
        return jsonify({'error': 'Code length must be between 1 and 10'}), 400
    
    # Get the unique values
    values = column.get('unique_values', [])
    
    if not values:
        return jsonify({'error': 'No values found in this column'}), 400
    
    # Try to determine what kind of values we're dealing with
    value_type = determine_value_type(values)
    
    # Get known mappings
    known_mappings = get_known_code_mappings()
    
    # Generate codes for each value
    generated_codes = []
    existing_codes = set()
    
    for value in values:
        code = generate_code_for_value(
            value, 
            code_length=code_length, 
            existing_codes=existing_codes,
            known_mappings=known_mappings
        )
        existing_codes.add(code)
        
        generated_codes.append({
            'value': value,
            'code': code
        })
    
    return jsonify({
        'success': True,
        'codes': generated_codes,
        'valueType': value_type
    })

@app.route('/api/get-dataset-columns/<int:index>', methods=['GET'])
def get_dataset_columns(index):
    """Get all columns from the uploaded dataset for import selection"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found. Please upload a file first.'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    
    if index < 0 or index >= len(session_data['columns']):
        return jsonify({'error': 'Invalid column index'}), 400
    
    # Load the dataframe
    df = load_uploaded_dataframe(session_data)
    if df is None:
        return jsonify({'error': 'Could not load dataset'}), 500
    
    # Get all column names
    columns = list(df.columns)
    
    # Get a preview of values from each column (first 5 non-null unique values)
    column_previews = {}
    for col in columns:
        unique_values = df[col].dropna().unique()
        preview = [str(v) for v in unique_values[:5]]
        column_previews[col] = preview
    
    return jsonify({
        'success': True,
        'columns': columns,
        'previews': column_previews
    })

@app.route('/api/import-codes/<int:index>', methods=['POST'])
def import_codes(index):
    """Import codes from a selected column in the dataset"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found. Please upload a file first.'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        cleanup_session_file(session['session_data_id'])
        session.pop('session_data_id', None)
        return jsonify({'error': 'Session expired. Please upload the file again.'}), 404
    
    if index < 0 or index >= len(session_data['columns']):
        return jsonify({'error': 'Invalid column index'}), 400
    
    column = session_data['columns'][index]
    
    if not column['is_codelist']:
        return jsonify({'error': 'This column is not a codelist'}), 400
    
    data = request.json
    code_column_name = data.get('codeColumn')
    
    if not code_column_name:
        return jsonify({'error': 'Code column name is required'}), 400
    
    # Load the dataframe
    df = load_uploaded_dataframe(session_data)
    if df is None:
        return jsonify({'error': 'Could not load dataset'}), 500
    
    if code_column_name not in df.columns:
        return jsonify({'error': f'Column "{code_column_name}" not found in dataset'}), 400
    
    # Get the unique values from the codelist column
    label_column_name = column['name']
    if label_column_name not in df.columns:
        return jsonify({'error': f'Label column "{label_column_name}" not found in dataset'}), 400
    
    # Create a mapping from unique values to their codes
    # Remove any null values and create a dictionary
    value_code_mapping = {}
    for idx, row in df[[label_column_name, code_column_name]].drop_duplicates().iterrows():
        label_value = row[label_column_name]
        code_value = row[code_column_name]
        
        # Skip if either is null
        if pd.isna(label_value) or pd.isna(code_value):
            continue
        
        # Convert to string
        label_str = str(label_value)
        code_str = str(code_value)
        
        value_code_mapping[label_str] = code_str
    
    # Get the unique values for this column
    values = column.get('unique_values', [])
    
    # Generate the codes list
    imported_codes = []
    missing_values = []
    
    for value in values:
        value_str = str(value)
        if value_str in value_code_mapping:
            imported_codes.append({
                'value': value,
                'code': value_code_mapping[value_str]
            })
        else:
            # Value not found in mapping - this shouldn't happen if the data is consistent
            missing_values.append(value)
            # Assign a placeholder or empty code
            imported_codes.append({
                'value': value,
                'code': ''
            })
    
    response = {
        'success': True,
        'codes': imported_codes
    }
    
    if missing_values:
        response['warning'] = f'{len(missing_values)} values could not be mapped to codes'
        response['missing_values'] = missing_values[:10]  # Return first 10 missing values
    
    return jsonify(response)

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """API endpoint to translate text using DeepL"""
    try:
        # Check if content type is JSON
        if not request.is_json:
            return jsonify({
                'success': False, 
                'error': f'Invalid content type. Expected application/json, got {request.content_type}'
            }), 400
        
        # Parse JSON safely
        try:
            data = request.get_json()
        except Exception as json_error:
            logging.exception(f"JSON parsing error: {json_error}")
            return jsonify({
                'success': False, 
                'error': 'Invalid JSON format'
            }), 400
        
        # Validate request data
        if not data:
            return jsonify({'success': False, 'error': 'Empty request data'}), 400
            
        # Check for different possible formats
        response_data = {'success': True}
        
        # Import translator only when needed
        try:
            from utils.translator import create_multilingual_text
        except ImportError as e:
            logging.error(f"Translator import error: {e}")
            return jsonify({'success': False, 'error': 'Translator module not available'}), 500
            
        source_lang = data.get('source_lang', 'DE')
        
        # Process 'text' field if present
        if 'text' in data:
            text = data.get('text', '')
            if text and text.strip():
                try:
                    # Detect language if not provided
                    if not source_lang or source_lang == 'auto':
                        source_lang = safe_detect_language(text)
                        
                    translations = create_multilingual_text(text, source_lang)
                    response_data['translations'] = translations
                except Exception as e:
                    logging.exception(f"Translation error: {e}")
                    return jsonify({
                        'success': False,
                        'error': 'Translation failed. Please try again or contact support.'
                    }), 500
        
        # Process 'title' field if present            
        if 'title' in data:
            title = data.get('title', '')
            if title and title.strip():
                try:
                    # Detect language if not provided
                    if not source_lang or source_lang == 'auto':
                        source_lang = safe_detect_language(title)
                        
                    title_translations = create_multilingual_text(title, source_lang)
                    if 'title' not in response_data:
                        response_data['title'] = title_translations
                except Exception:
                    pass
        
        # Process 'description' field if present
        if 'description' in data:
            description = data.get('description', '')
            if description and description.strip():
                try:
                    # Detect language if not provided
                    if not source_lang or source_lang == 'auto':
                        source_lang = safe_detect_language(description)
                        
                    desc_translations = create_multilingual_text(description, source_lang)
                    if 'description' not in response_data:
                        response_data['description'] = desc_translations
                except Exception:
                    pass
        
        # Process 'keywords' field if present
        if 'keywords' in data:
            keywords = data.get('keywords', '')
            if keywords and keywords.strip():
                try:
                    # Detect language if not provided
                    if not source_lang or source_lang == 'auto':
                        source_lang = safe_detect_language(keywords)
                        
                    keywords_translations = create_multilingual_text(keywords, source_lang)
                    if 'keywords' not in response_data:
                        response_data['keywords'] = keywords_translations
                except Exception:
                    pass
        
        # If none of the expected fields were processed
        if len(response_data) <= 1:  # Only has 'success' key
            fields = list(data.keys())
            return jsonify({
                'success': False, 
                'error': f'No valid text fields to translate. Available fields: {fields}'
            }), 400
            
        return jsonify(response_data)
            
    except Exception as e:
        logging.exception(f"Translation request error: {e}")
        return jsonify({
            'success': False,
            'error': 'Translation failed. Please try again or contact support.'
        }), 500

@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    """Verify token and return associated agencies"""
    # Try to get data from both form and JSON
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
    
    if not data or 'token' not in data:
        return jsonify({'success': False, 'error': 'Token is required'}), 400
    
    token = data['token']
    
    # Clean the token by removing Bearer prefix if present
    clean_token = token.strip()
    if clean_token.upper().startswith('BEARER '):
        clean_token = clean_token[7:].strip()
    
    # Extract user email and agency identifier from token
    user_email = None
    token_agency_identifier = None
    token_agency_label = None
    try:
        # JWT tokens are in format: header.payload.signature
        token_parts = clean_token.split('.')
        if len(token_parts) >= 2:
            # Decode the payload part
            payload = token_parts[1]
            payload += '=' * ((4 - len(payload) % 4) % 4)  # Add padding
            decoded_payload = base64.b64decode(payload).decode('utf-8')
            payload_data = json.loads(decoded_payload)
            
            # Extract email
            user_email = payload_data.get('email', payload_data.get('upn', payload_data.get('preferred_username')))
            
            # Extract agency identifier and label from token
            # Format: "6517609\\i14y-test-organisation"
            agency_strings = payload_data.get('agencies', [])
            if agency_strings and len(agency_strings) > 0:
                agency_str = agency_strings[0]
                parts = agency_str.split('\\')
                if len(parts) >= 2:
                    token_agency_identifier = parts[0]  # Get the identifier part (e.g., "6517609")
                    token_agency_label = parts[1]  # Get the label part (e.g., "i14y-test-organisation")
                elif len(parts) == 1:
                    token_agency_identifier = parts[0]  # Only identifier provided
    except Exception as e:
        logging.exception(f"Error extracting agency from token: {e}")
        pass
    
    # Fetch agencies from the I14Y API (authoritative source)
    agencies = fetch_user_agencies(clean_token)
    
    if not agencies:
        return jsonify({
            'success': False, 
            'error': 'Could not extract agencies from the provided token. Please verify your token is correct.'
        }), 400
    
    # Find the preselected agency by matching the identifier or label from token with agencies
    preselected_agency = None
    if token_agency_identifier or token_agency_label:
        print(f"Looking for agency - Identifier: {token_agency_identifier}, Label: {token_agency_label}")
        print(f"Available agencies: {[{'id': a['identifier'], 'name': a.get('name', {})} for a in agencies]}")
        
        for agency in agencies:
            agency_id = agency.get('identifier')
            agency_name = agency.get('name', {})
            
            # Try to match the label from token against the agency identifier (most common case)
            # In JWT format "6517609\\i14y-test-organisation", the second part is the actual agency ID
            if token_agency_label and agency_id == token_agency_label:
                preselected_agency = agency_id
                print(f"Matched agency by label to identifier: {agency_id}")
                break
            
            # Also try to match by the first part (identifier) in case format is different
            if token_agency_identifier and agency_id == token_agency_identifier:
                preselected_agency = agency_id
                print(f"Matched agency by identifier: {agency_id}")
                break
            
            # Finally, try to match by label against the agency name in different languages
            if token_agency_label and isinstance(agency_name, dict):
                # Check all language variants
                for lang_name in agency_name.values():
                    if isinstance(lang_name, str) and lang_name.lower() == token_agency_label.lower():
                        preselected_agency = agency_id
                        print(f"Matched agency by name: {agency_id} ({lang_name})")
                        break
            
            if preselected_agency:
                break
        
        if not preselected_agency:
            print(f"No matching agency found for identifier '{token_agency_identifier}' or label '{token_agency_label}'")
    
    # Return the agencies, email, and preselected agency
    response_data = {
        'success': True,
        'agencies': agencies,
        'user_email': user_email
    }
    
    if preselected_agency:
        response_data['preselected_agency'] = preselected_agency
    
    return jsonify(response_data)

@app.route('/api/save-submission-info', methods=['POST'])
def save_submission_info():
    """Save token and email information to session for reuse"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        return jsonify({'error': 'Session expired'}), 404
    
    data = request.get_json() or {}
    
    # Save submission info
    if 'submission_info' not in session_data:
        session_data['submission_info'] = {}
    
    if data.get('token'):
        session_data['submission_info']['token'] = data['token']
    if data.get('responsible_person'):
        session_data['submission_info']['responsible_person'] = data['responsible_person']
    if data.get('responsible_deputy'):
        session_data['submission_info']['responsible_deputy'] = data['responsible_deputy']
    if data.get('agency'):
        session_data['submission_info']['agency'] = data['agency']
    
    try:
        save_session_data(session_data, session_id=session['session_data_id'])
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving submission info: {e}")
        return jsonify({'error': 'Failed to save submission info'}), 500

@app.route('/api/get-submission-info', methods=['GET'])
def get_submission_info():
    """Retrieve saved token and email information from session"""
    if 'session_data_id' not in session:
        return jsonify({'error': 'No session found'}), 404
    
    session_data = load_session_data(session['session_data_id'])
    if not session_data:
        return jsonify({'error': 'Session expired'}), 404
    
    submission_info = session_data.get('submission_info', {})
    
    return jsonify({
        'success': True,
        'submission_info': submission_info
    })

if __name__ == '__main__':
    print("Starting I14Y AutoImport application...")
    print("Server will be available at: http://127.0.0.1:5000")
    # Bind all interfaces: required in containerized deployments where the platform's ingress reaches the app via the container network.
    app.run(host='0.0.0.0', port=5000)  # nosec B104  # nosemgrep: python.flask.security.audit.app-run-param-config.avoid_app_run_with_bad_host