import bcrypt
import getpass

# --- 1. Get the password securely ---
# Using getpass is more secure as it doesn't show the password on the screen
try:
    password = getpass.getpass("Enter your admin password: ")
except Exception as error:
    print('ERROR', error)
    sys.exit(1)


# --- 2. Generate the hash ---
# The password needs to be encoded to UTF-8 bytes
hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


# --- 3. Decode the hash to a string for storage ---
# The result is a byte string, so we decode it to a regular string to store it
hashed_string = hashed_bytes.decode('utf-8')

print("\n✅ New Hashed Password Generated Successfully!\n")
print("Copy this entire line and update the ADMIN_PASSWORD secret in Doppler:")
print("----------------------------------------------------------------------")
print(hashed_string)
print("----------------------------------------------------------------------")
