## installer

```powershell
cd D:\_work\projects\autoTraining\Autotrain_g
py -3.11 -m venv .venv-napari
.\.venv-napari\Scripts\pip install -r tools\napari_seg\requirements.txt

.\.venv-napari\Scripts\python tools\napari_seg\annotate.py `
    --images D:\data\raw_images `
    --out    D:\data\out\data `
    --classes partA,partB
```


other command
```powershell
cd D:\_work\projects\autoTraining\Autotrain_g
py -3.11 -m venv .venv-napari
.\.venv-napari\Scripts\pip install -r tools\napari_seg\requirements.txt

.\.venv-napari\Scripts\python tools\napari_seg\annotate.py `
    --images G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm `
    --out    G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm_out `
    --classes SAF_Ru_L,SAF_Ru_R,MgO_L,MgO_R,MgO_C,Non_mag,Milling_L,Milling_R,Leveling,Block_U,Block_D
```

## check how filenames are rewritten

Anything outside ASCII `[A-Za-z0-9._-]` becomes `_`, because PaddleSeg splits
`train.txt` lines on whitespace. Collisions get a `_2`, `_3`, ... suffix.

```powershell
.\.venv-napari\Scripts\python tools\napari_seg\annotate.py `
    --images G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm `
    --out    G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm_out `
    --print-names --no-gui
```

## grow the dataset with model predictions (pseudo-labelling)

Predict with the trained model first. Use `pseudo_color_prediction/`, NOT
`added_prediction/` (the latter is the image blended with the mask, not a label
map -- the importer refuses it).

```powershell
cd D:\_work\projects\autoTraining\pd\PaddleSeg
python tools\predict.py `
    --config <config>.yml `
    --model_path output\best_model\model.pdparams `
    --image_path G:\datasets\TEM\new_batch `
    --save_dir   G:\datasets\TEM\new_batch_pred
```

Import the predictions and open napari to review/correct them:

```powershell
cd D:\_work\projects\autoTraining\Autotrain_g
.\.venv-napari\Scripts\python tools\napari_seg\annotate.py `
    --images G:\datasets\TEM\new_batch `
    --out    G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm_out `
    --import-masks G:\datasets\TEM\new_batch_pred\pseudo_color_prediction
```

Existing masks are kept unless `--overwrite` is passed. Add `--no-gui` to import
without opening the annotator.