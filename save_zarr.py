import spatialdata_io as sio

xenium_path = "./data/Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/"

sdata = sio.xenium(xenium_path, gex_only=False, morphology_focus=False, cells_boundaries=False, nucleus_boundaries=False, cells_labels=False, nucleus_labels=False, cells_as_circles=True)
sdata.write(xenium_path + 'data.zarr')