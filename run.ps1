param (
    [string]$camera = "0"
)

# Activate the virtual environment
& "venv\Scripts\Activate"

# Run main.py with the specified camera argument
python main.py --camera $camera
