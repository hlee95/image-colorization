#!/bcn/sh -e
#
# rc.local
#
# This script is executed at the end of each multiuser runlevel.
# Make sure that the script will "exit 0" on success or any other
# value on error.
#
# In order to enable or disable this script just change the execution
# bits.
#
# By default this script does nothing.

# This is the startup script that is run when our AWS instance boots up.
# Always store the current time into persistent file startup.txt.
# Move into the correct directory then run convolutional.py.
# Redirect output into output.txt. Can use tail -f to view output.

echo "Starting up. Current time:" >> /home/ubuntu/startup.txt;
date >> /home/ubuntu/startup.txt;
cd /home/ubuntu/image-colorization/sandbox/mnist_copy;
python convolutional.py >> output.txt;
exit 0;
