# Test: Record a patient interaction (patient-services portal)

## About
Mirrors a real patient-support-programme flow: sign in, open the programme, pick one patient out of
near-identical names, record an interaction through a sidebar with three kinds of custom dropdown, a
calendar date picker and an HH / MM time popover, then check the interaction log.

## How to run
Start the demo app (python -m demo.app), then run this file with the run variable care_email and the
secret care_password (the demo account is listed in docs/windows-local-setup.md). Credentials are never
written into the test: the e-mail is a run variable and the password comes from the encrypted secret store.

## Steps
http://127.0.0.1:8800/care/sign-in
Enter {{care_email}} as Email
Enter {{care_password}} as Password
Click on Login button
Click on Patient Support Program
Select the Patient named Abc Xy
Click on Interaction tab
Click on Record Interaction to open sidebar
Click on Theme Box and select Initial Call
Click on Status box and select Not reachable
Click on Call Direction and select Outbound
Click on Start Date Selection and select 09 June 2026
Select Start Time and set the first field as 23 HH and second field as 15 MM
Select Record Interaction
Check whether new Interaction for patient Abc is added on the Interaction Logs tab
