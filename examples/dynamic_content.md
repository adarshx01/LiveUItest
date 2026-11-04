# Test: Dynamic and lazy content

URL: http://127.0.0.1:8800/widgets/dynamic

## Steps
1. Wait for the loading spinner to disappear
2. Click Dismiss Password changed
3. Verify that "Password changed" is not displayed
4. Click Mark all as read
5. Verify that "All caught up" is displayed
6. Go to http://127.0.0.1:8800/widgets/infinite
7. Scroll down until "Item 55" is visible
8. Verify that "Item 55" is displayed
9. Go to http://127.0.0.1:8800/widgets/icons
10. Click the settings icon
11. Verify that "Settings opened" is displayed
12. Click the star icon
13. Verify that "Added to favorites" is displayed
