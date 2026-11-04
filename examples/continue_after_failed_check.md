# Test: A failed check does not stop independent steps

## About
With AITEST_CONTINUE_ON_FAILURE=assertions (the default) a failed verification is reported and the run
carries on with the remaining steps; a failed action still skips the steps that depend on it.
Expected outcome: FAILED (the wrong status check), with every later step executed.

## Steps
http://127.0.0.1:8800/care/sign-in
Enter {{care_email}} as Email
Enter {{care_password}} as Password
Click on Login button
Click on Patient Support Program
Select the Patient named Abc Xy
Verify that "Status: Discharged" is displayed
Click on Interaction Logs tab
Verify that "No interactions recorded yet" is displayed
