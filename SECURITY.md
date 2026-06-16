# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | ✅ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately by emailing **shaik.izaaz009@gmail.com** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive an acknowledgment within **48 hours**. If confirmed, a patch will be released promptly.

## Security Best Practices

When deploying this project:

- Never commit real API keys or secrets to version control
- Use `.env` files (excluded from git) for local development  
- Use GitHub Secrets or a dedicated secrets manager in production
- Keep all dependencies up to date — run `pip install --upgrade -r requirements.txt` regularly
- Enable Dependabot alerts on your fork

## Scope

The following are in scope for security reports:

- Authentication or authorization bypass
- Remote code execution
- SQL/command injection
- Sensitive data exposure
- Denial of service vulnerabilities

## Hall of Fame

Responsible disclosures will be credited here (with your permission).
