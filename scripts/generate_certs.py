#!/usr/bin/env python3
import os
import sys
import datetime
import ipaddress

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
except ImportError:
    print("Error: 'cryptography' library is required to run this script.")
    print("Please install it using: pip install cryptography")
    sys.exit(1)

def generate_self_signed_cert(cert_dir, domain="10.7.31.114"):
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "server.crt")
    key_path = os.path.join(cert_dir, "server.key")

    print(f"Generating self-signed SSL certificate for domain: {domain}")
    print(f"Saving certificate to: {cert_path}")
    print(f"Saving private key to: {key_path}")

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Prepare Subject and Issuer Name
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Classroom IoT"),
        x509.NameAttribute(NameOID.COMMON_NAME, domain),
    ])

    # Subject Alternative Names (SAN)
    sans = [
        x509.DNSName("localhost"),
        x509.DNSName(domain),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    # If the domain is an IP address, add it as IPAddress SAN as well
    try:
        ip = ipaddress.ip_address(domain)
        sans.append(x509.IPAddress(ip))
    except ValueError:
        pass

    # Build Certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(list(set(sans))),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Save Private Key
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Save Certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print("Success: Generated self-signed certificate and key.")

if __name__ == "__main__":
    # Base directory is parent of scripts directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cert_dir = os.path.join(base_dir, "certs")
    generate_self_signed_cert(cert_dir)
