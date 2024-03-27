from preprocessing.io import qptiff_to_zarr

def test_qptiff_to_zarr():
    # Step 1: Define the input file and output directory
    input_file = "/gpfs/scratch/jt3545/projects/CODEX/data/kidney/AMP_1158.qptiff"
    #input_file = "/gpfs/scratch/jt3545/projects/CODEX/data/kidney/AMP_1156.qptiff"
    output_root = "/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/data/"
    
    # Step 2: Convert the QPTIFF file to a Zarr Array
    zarr_array = qptiff_to_zarr(input_file, output_root)

if __name__ == '__main__':
    test_qptiff_to_zarr()
