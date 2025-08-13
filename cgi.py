# cgi.py
# Minimal replacement for removed cgi module in Python 3.13+
# Only includes parse_header, which is used by httpx/googletrans

def parse_header(line):
    parts = line.split(";", 1)
    key = parts[0].strip()
    params = {}
    if len(parts) > 1:
        for item in parts[1].split(";"):
            if "=" in item:
                name, value = item.strip().split("=", 1)
                params[name.lower()] = value.strip('"')
    return key, params
