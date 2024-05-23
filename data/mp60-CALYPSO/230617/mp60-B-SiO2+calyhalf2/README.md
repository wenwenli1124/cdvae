# MP60-CALYPSO

MP60-CALYPSO contains 670979 locally stable structures with pressure labeled from Materials Project and CALYPSO community.
Please refer to XXX for more details.

The data will soon be available at [CALYPSO website](http://calypso.cn).

~~Download the train-val-test splited files from~~

There contains three feather files which can be read by pandas.

```python
import pandas as pd

trn_df = pd.read_feather("train.feather")
val_df = pd.read_feather("val.feather")
tst_df = pd.read_feather("test.feather")
```
