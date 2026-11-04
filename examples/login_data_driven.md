# Test: Verify user login (data-driven)

URL: http://127.0.0.1:8800

## Test Data
- username: abc
- display_name: abc

## Steps
1. Open the login page.
2. Click on the Login button.
3. Enter username as {{username}}.
4. Enter password as {{test_password}}.
5. Click on the Sign In button.
6. Verify that the user is redirected to the dashboard.
7. Verify that the dashboard displays "{{display_name}}".
