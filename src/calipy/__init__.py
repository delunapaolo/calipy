import sys

# ------------------------------------------------------------------------------
# Print error messages from Qt
# Back up the reference to the exceptionhook
# Back up the reference to the exception hook
sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    sys.stdout.write(exctype, value, traceback)
    sys.stdout.write('\n')
    sys.stdout.flush()
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)
