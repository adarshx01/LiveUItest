# Test: Registration rejects mismatched passwords

URL: http://127.0.0.1:8800/register

## Steps
1. Enter Full name as Sam Tester
2. Enter Email as sam@example.com
3. Enter password as Secret123!
4. Enter confirm password as Different99!
5. Select Business
6. Enter Company name as Acme Labs
7. Accept the terms and conditions
8. Click Create account
9. Verify that "Passwords do not match" is displayed
10. Verify that "Account created" is not displayed
