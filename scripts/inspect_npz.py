import numpy as np
import argparse
import os
import sys

def inspect_npz(file_path):
    """
    Inspects and prints the summary of a .npz file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Load the npz file
        data = np.load(file_path, allow_pickle=True)
        print(f"\nInspecting: {os.path.abspath(file_path)}")
        print("=" * 100)
        
        # Header
        print(f"{'Key':<30} {'Shape':<20} {'Dtype':<10} {'Min / Max / Mean (if numeric)'}")
        print("-" * 100)

        keys = sorted(data.files)
        for key in keys:
            try:
                arr = data[key]
                shape_str = str(arr.shape)
                dtype_str = str(arr.dtype)
                
                stats = ""
                # Check if it's a numeric type for statistics
                if np.issubdtype(arr.dtype, np.number):
                    if arr.size > 0:
                        stats = f"{np.min(arr):.4f} / {np.max(arr):.4f} / {np.mean(arr):.4f}"
                    else:
                        stats = "Empty"
                # Check for string types
                elif arr.dtype.kind in {'U', 'S', 'O'}: 
                     if arr.size > 0:
                         # Print first element as example, truncate if too long
                         example = str(arr.flatten()[0])
                         if len(example) > 40:
                             example = example[:37] + "..."
                         stats = f"Example: {example}"
                     else:
                         stats = "Empty"
                else:
                    stats = "N/A"

                print(f"{key:<30} {shape_str:<20} {dtype_str:<10} {stats}")
            except Exception as e:
                print(f"{key:<30} {'Error accessing data':<20} {str(e)}")
        
        print("=" * 100)
        print(f"Total keys: {len(keys)}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a .npz file.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the .npz file")
    args = parser.parse_args()

    inspect_npz(args.file_path)
