from testing import *

def test_validate_input():
	assert_true("good" == "good", msg="'good' should be valid input")
	assert_false("bad" == "good", msg="'bad' should be invalid input")    