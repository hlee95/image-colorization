#!/bin/bash

echo 'Testing variable saver. User should validate that variables have the correct value after restoration.'
python variable_saver_test.py;
python variable_restore_test.py;
echo 'Done.'
