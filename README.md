1. Install necessary requirements:

       pip install -r requirements.txt

2. Run the demo:

       python demo.py --output results.csv

   The demo will take **a lot of time to run** (depending on the ranges specified in the script, currently on the order of 10 minutes). **It cannot be properly stopped by `SIGINT`**, you would have to kill the Python process if you need to stop the demo. The demo will display a `matplotlib` plot in a separate window (blocking until the window is closed), and it will also save the data to the file `results.csv`.

    *Note: The demo is so slow as a result of implementation details which are hard to optimize. See the note in `demo.py` for details.*

   The saved data can later be displayed without running the trials again:

       python demo.py --input results.csv
