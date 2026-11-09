# Test: WebForms registration - paired selects, Range (%) grid, legacy inputs, confirm dialog

URL: http://127.0.0.1:8800/registry/register

## Steps
1. Select "2026/11" from the "Master Year Month" dropdown
2. Select "2026/04" from the first "Data period" dropdown
3. Select "2026/09" from the second "Data period" dropdown
4. Enter Study Code as "abc-123"
5. Enter Quantity as 12000
6. Enter Delivery Date as 2026/12/15
7. Click on the input field "1", clear the existing value, and enter "01" below the "Horizontal" dropdown
8. Click on the input field below "01" and enter "100"
9. Click on the second row input field under column "3" and enter "25.5"
10. Select "Country" in the "Available columns" list
11. Click on the ">" button
12. Verify that the "Show subtotal" checkbox is enabled
13. Check "Show subtotal"
14. Verify that "Approved" is shown in green
15. Verify that "Rejected" is displayed in red
16. Click on the "Register" button
17. Click "OK" on the confirmation popup
18. Verify that "Registration completed" is displayed
19. Verify that "Period: 2026/04 ～ 2026/09" is displayed
20. Verify that "Study Code: ABC-123" is displayed
21. Verify that "Quantity: 12,000" is displayed
22. Verify that "Delivery: 2026/12/15" is displayed
23. Verify that "Range: 01=100.000, 3=25.500" is displayed
24. Verify that "Columns: Country (with subtotal)" is displayed
