from preprocessing.tile import gen_tiles

if __name__ == '__main__':
    slide_path = "/gpfs/scratch/jt3545/projects/CODEX/analysis/kidney/data/AMP_1156/data.zarr"
    tile_size = 128
    gen_tiles(slide_path, tile_size=tile_size)
