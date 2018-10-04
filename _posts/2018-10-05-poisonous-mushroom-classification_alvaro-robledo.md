---
layout: post
title: Determining the edibility of wild mushrooms
featured-img: wild-mushrooms
summary: Can we use Machine Learning to determine whether a mushroom is edible or not, just by looking at it's physical characteristics? Let's find out.
categories: [Machine Learning, Classification]
---

# Determining the edibility of wild mushrooms

*Note: the following content can also be viewed as a [Jupyter notebook](http://nbviewer.jupyter.org/github/alvarorobledo/Poisonous-Mushroom-Classification/blob/master/poisonous-mushroom-classification_alvaro-robledo.ipynb), or in my [Github repository](https://github.com/alvarorobledo/Poisonous-Mushroom-Classification)*

Is it possible to tell whether a mushroom is edible ot not, just by looking at it's physical characteristics? We will explore this question using [this dataset](https://www.kaggle.com/uciml/mushroom-classification) from the UCI Machine Learning.


### Dataset description:

This dataset contains 8124 entries corresponding to 23 species of gilled mushrooms from North America. Each species is identified as definitely edible (e), definitely poisonous (p), or of unknown edibility and not recommended (also p). Each entry has 22 features related to the physical characteristics of the mushroom. The feature labels are explained in the file labels.txt. (Data source: The Audubon Society Field Guide to North American mushrooms).

### Importing all libraries


```python
%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Dataset: loading and initial inspection


```python
df = pd.read_csv('dataset.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>...</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>e</td>
      <td>x</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>b</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>w</td>
      <td>v</td>
      <td>d</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>...</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 23 columns</p>
</div>



We notice that the column _veil-type_ has only 1 unique value - that is, all 8124 mushroom instances have the same veil-color.

It thus becomes an irrelevant feature, so we proceed to remove it


```python
df.drop(['veil-type'], axis=1, inplace=True)
```

### Converting categorical data to numerical

Most Machine Learning algorithms require _numerical features_. However, our dataset is composed of _categorical features_. We now proceed to convert these to numerical.

#### Label Encoding

A typical approach is to perform _Label Encoding_. This is nothing more than just assigning a number to each category, that is:

(cat_a, cat_b, cat_c, etc.) → (0, 1, 2, etc.)

This technique works:
* When the features are binary (only have 2 unique values)
* When the features are _ordinal categorical_ (that is, when the categories can be ranked). A good example would be a feature called _t-shirt size_ with 3 unique values _small_, _medium_ or _large_, which have an intrinsic order.

__However__, in our case, only some of our features have 2 unique values (most of them have more), and none of them are _ordinal categorical_ (in fact they they are _nominal categorical_, which means they have no intrinsic order).

Therefore, we will only apply Label Encoding to those features with a binary set of values:




```python
for col in df.columns:
    if len(df[col].value_counts()) == 2:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>1</td>
      <td>p</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>1</td>
      <td>a</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>1</td>
      <td>l</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>1</td>
      <td>p</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>0</td>
      <td>n</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



We can see how it has converted some of the features to values of 0 or 1. More importantly, our labels (the _class_ column) are now 0=e, and 1=p.

#### One Hot Encoding

For the remaining features, we can use a technique called One Hot Encoding.

Essentially, this consists on creating a new binary feature representing each category. For instance, from the feature _cap surface_, which has 4 unique values (f, g, y and s), we create 4 binary features (cap_surface_f, cap_surface_g, cap_surface_y and cap_surface_s) indicating whether the category they represent was indeed that one or not. This means that, for any given instance (row), we will have exactly one of these 4 features equal to 1, and the other 3 equal to 0.

One Hot Encoding is really simple to perform with the _pandas_ package:


```python
df = pd.get_dummies(df)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>bruises</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>stalk-shape</th>
      <th>cap-shape_b</th>
      <th>cap-shape_c</th>
      <th>cap-shape_f</th>
      <th>cap-shape_k</th>
      <th>...</th>
      <th>population_s</th>
      <th>population_v</th>
      <th>population_y</th>
      <th>habitat_d</th>
      <th>habitat_g</th>
      <th>habitat_l</th>
      <th>habitat_m</th>
      <th>habitat_p</th>
      <th>habitat_u</th>
      <th>habitat_w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 112 columns</p>
</div>



### Separating labels from features

X will now contain our features, and y our labels (0 for edible and 1 for poisonous/unknown)


```python
y = df['class'].to_frame()
X = df.drop('class', axis=1)
```


```python
y.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bruises</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>stalk-shape</th>
      <th>cap-shape_b</th>
      <th>cap-shape_c</th>
      <th>cap-shape_f</th>
      <th>cap-shape_k</th>
      <th>cap-shape_s</th>
      <th>...</th>
      <th>population_s</th>
      <th>population_v</th>
      <th>population_y</th>
      <th>habitat_d</th>
      <th>habitat_g</th>
      <th>habitat_l</th>
      <th>habitat_m</th>
      <th>habitat_p</th>
      <th>habitat_u</th>
      <th>habitat_w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 111 columns</p>
</div>



### Standardising our features

It is generally considered a _good practice_ to standardise our features (convert them to have zero-mean and unit variance). Most of the times, the difference will be small, but, in any case, it still never hurts to do so.


```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Creating training and sets sets

We will separate our data into a training set (70%) and a test set (30%). This is a very standard approach in Machine Learning.

The _stratify_ option ensures that the ratio of edible to poisonois mushrooms in our dataset remains the same in both training and test sets. The *random_state* parameter is simply a seed for the algorithm to use (if we didn't specify one, it would create different training and test sets every time we run it)


```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=19)
```

### Logistic Regression

Since this is now a supervised learning binary classification problem, it makes perfect sense to start by running a simple _logistic regression_.

A logistic regression simply predicts the probability of an instance (row) belonging to the default class, which can then be snapped into a 0 or 1 classification. Off we go.


```python
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
y_pred_test = logreg.predict(X_test)
print('Accuracy of Logistic Regression classifier on the test set: {:.2f}'.format(accuracy_score(y_test, y_pred_test)))
```

    Accuracy of Logistic Regression classifier on the test set: 1.00
    

It seems like the logistic regression achieved the maximum accuracy possible: 100%

I have to admit that this made me go back and check my code and logical reasoning a couple times. But no, it simply means that the given features are a really good indicator of the edibility of mushrooms.

Still, we should run the logistic regression again, but this time using _cross-validation_, to ensure that we are not overfitting the data. A simple 10-fold cross validation should do.



```python
scores = cross_val_score(logreg, X_train, y_train.values.ravel(), cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=19), scoring='accuracy')
print('Accuracy of Logistic Regression classifier using 10-fold cross-validation: {}'.format(scores.mean()))
```

    Accuracy of Logistic Regression classifier using 10-fold cross-validation: 0.9997655334114889
    

This time it doesn't achieve the perfect score, but it's pretty damn close.

Thus, it seems like the relationship between the features and the edibility of the mushrooms is __highly linear__. There is really no point in trying other models different from this logistic regression.

What we can do is investigate what are the most immportant features in deciding whether a mushroom is edible or not.

### Most relevant features


```python
features_coeffs = pd.DataFrame(logreg.coef_, columns=X.columns, index=['coefficients'])
features_coeffs.sort_values('coefficients', axis=1, ascending=False, inplace=True)
features_coeffs.T.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>odor_p</th>
      <td>1.301698</td>
    </tr>
    <tr>
      <th>odor_c</th>
      <td>1.248420</td>
    </tr>
    <tr>
      <th>odor_f</th>
      <td>1.215397</td>
    </tr>
    <tr>
      <th>spore-print-color_r</th>
      <td>1.186042</td>
    </tr>
    <tr>
      <th>spore-print-color_h</th>
      <td>1.101121</td>
    </tr>
  </tbody>
</table>
</div>




```python
features_coeffs.T.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gill-spacing</th>
      <td>-0.777519</td>
    </tr>
    <tr>
      <th>odor_a</th>
      <td>-0.783296</td>
    </tr>
    <tr>
      <th>spore-print-color_n</th>
      <td>-0.819762</td>
    </tr>
    <tr>
      <th>odor_l</th>
      <td>-0.827127</td>
    </tr>
    <tr>
      <th>odor_n</th>
      <td>-1.719140</td>
    </tr>
  </tbody>
</table>
</div>



Interesting. Seems like _odor_ and _spore-print-color_ play an important role in deciding whether a mushroom is edible or not. Let's confirm this:


```python
def plot_features_containing(feature_name):
    categories = X.columns[X.columns.str.contains(feature_name)]
    edible_num = []
    poisonous_num = []
    for cat in categories:
        y[X[cat]==0]
        edible_count = sum((y[X[cat]==1]==0).values[:,0])
        poisonous_count = sum(X[cat]==1) - edible_count
        edible_num.append(edible_count)
        poisonous_num.append(poisonous_count)
    odor_df = pd.DataFrame(index=categories, columns=['edible', 'poisonous'])
    odor_df.edible = edible_num
    odor_df.poisonous = poisonous_num
    odor_df.plot(x=odor_df.index, kind='bar')
plot_features_containing('odor')
```


![Mushrooms by odor - graph](https://alvarorobledo.com/assets/img/posts_contents/mushrooms-by-odor.png "Mushrooms by odor - graph")


odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

Very interesting! Seems like, at least in our dataset set:
* All mushrooms with an almond or anise odor are edible
* All mushrooms with a creosote, fishy, musty, pungent or spicy odor are poisonous (or unknown edibility)
* Most mushrooms with no odor are edible. But not all of them!

Of course, this is just what our dataset tells us. It doesn't necessarily mean that any new mushroom we find out there will obey these rules 


```python
plot_features_containing('spore-print-color')
```


![Mushrooms by spore print color - graph](https://alvarorobledo.com/assets/img/posts_contents/mushrooms-by-spore-print-color.png "Mushrooms by spore print color - graph")

For _spore-print-color_ we have quite a similar picture, although perhaps not as extreme as with _odor_. This is what we expected, since these are the 2 features with the highest coefficients in our logistic regression.

In fact, if we do the same for a feature different from these two, the distribution will probably not be as extreme as for these last 2.

Let's check this.


```python
plot_features_containing('cap-color')
```


![Mushrooms by cap color - graph](https://alvarorobledo.com/assets/img/posts_contents/mushrooms-by-cap-color.png "Mushrooms by cap color - graph")


Indeed, we see a much more balanced distribution, which suggests that cap-color does not play such an important role in determining the edibility of a mushroom.

## Conclusion

* We fitted a logistic regression model and achieved near perfect accuracy, so there was no need to try with more complex models.

* Our algorithm identified specific traits (particularly regarding _odor_) that seem to heavily influence the chance that a mushroom is edible or not.

* Even though experts have determined that is that there is no simple set of rules to determine whether a mushroom is edible or not, it seems like with this algorithm we can get pretty close.

Nevertheless, it is important to keep in mind that __these results apply only to this dataset__, and don't necessarily mean that there aren't any mushrooms out there which don't follow these rules.

So, if you're ever lost and stranded in a forest, don't attempt to eat anything just because a machine tells you to do so! _Stay safe out there_.
