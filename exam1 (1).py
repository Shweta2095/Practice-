{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e4492988",
   "metadata": {},
   "source": [
    "# Programming in Python\n",
    "## Exam: September 19, 2022\n",
    "\n",
    "You can solve the exercises below by using standard Python 3.10 libraries, NumPy, Matplotlib, Pandas, PyMC3.\n",
    "You can browse the documentation: [Python](https://docs.python.org/3.10/), [NumPy](https://numpy.org/doc/stable/user/index.html), [Matplotlib](https://matplotlib.org/stable/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html), [PyMC3](https://docs.pymc.io/en/v3/index.html).\n",
    "You can also look at the [slides of the course](https://homes.di.unimi.it/monga/lucidi2122/pyqb00.pdf) or your code on [GitHub](https://github.com).\n",
    "\n",
    "**It is forbidden to communicate with others.** \n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5856860",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd  # type: ignore\n",
    "import matplotlib.pyplot as plt # type: ignore\n",
    "import pymc3 as pm   # type: ignore\n",
    "import arviz as az   # type: ignore"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3299c3ae",
   "metadata": {},
   "source": [
    "### Exercise 1 (max 2 points)\n",
    "\n",
    "The file [butterfly_data.csv](./butterfly_data.csv) (source: https://doi.org/10.13130/RD_UNIMI/5ZXGIV) contains data about a population of butterflies.\n",
    "\n",
    "Load the data in a pandas dataframe; be sure the columns `organic` and `alternate_management` have the `bool` dtype.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9aa65d9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import csv\n",
    "with open('butterfly_data.csv') as csv_file:\n",
    "    csv_read=csv.reader(csv_file, delimiter=',')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a16203f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('butterfly_data.csv')\n",
    "# Convert the columns to bool dtype\n",
    "df['organic'] = df['organic'].astype(bool)\n",
    "df['alternate_management'] = df['alternate_management'].astype(bool)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4dff3d2a",
   "metadata": {},
   "source": [
    "### Exercise 2 (max 5 points)\n",
    "\n",
    "Make a figure with a scatterplot of the `x` and `y` values; each point should be colored according its `subarea`. Use a proper title and a legend (Hint: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "db704939",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Get the unique values of the subarea column\n",
    "subareas = df['subarea'].unique()\n",
    "\n",
    "# Define a color map\n",
    "color_map = plt.cm.get_cmap('viridis')\n",
    "\n",
    "# Create a scatterplot with different colors for each subarea\n",
    "for subarea in subareas:\n",
    "    subarea_data = df[df['subarea'] == subarea]\n",
    "    plt.scatter(subarea_data['x'], subarea_data['y'], label=subarea, c=color_map(np.random.rand(1)))\n",
    "\n",
    "# Add a title and a legend\n",
    "plt.title(\"Scatterplot of x and y by subarea\")\n",
    "plt.legend()\n",
    "\n",
    "# Show the plot\n",
    "plt.show()\n",
    " \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1901ada0",
   "metadata": {},
   "source": [
    "### Exercise 3 (max 7 points)\n",
    "\n",
    "Define a function `distance` that takes two points in a 2D Cartesian plane and returns the Euclidean distance ($d = \\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$) between them. \n",
    "\n",
    "To get the full marks, you should declare correctly the type hints and add a test within a doctest string."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "9fbdc987",
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "from typing import Tuple #for typehints and docstring\n",
    "\n",
    "def distance(x1:'float',y1:'float',x2:'float',y2:'float')-> float:\n",
    "    \n",
    "    \"\"\"The function distance takes two points ina 2D Cartesian plane with co-ordinates\n",
    "    (x1,y1) and (x2,y2) and returns the distance between them.\n",
    "    \n",
    "    >>>distance (2,2,4,4)\n",
    "    '2.83 is the distance'\n",
    "    \n",
    "    \"\"\"\n",
    "    \n",
    "    d=(x1-x2)**2+(y1-y2)**2\n",
    "    return math.sqrt(d)\n",
    "\n",
    "\n",
    "    #return f'{math.sqrt(d):.2f} is the distance'\n",
    "    #return round(math.sqrt(d), 2) to round to 2 decimals\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "91f3cdef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.8284271247461903"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "distance (2,2,4,4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ebb3c8b2",
   "metadata": {},
   "source": [
    "### Exercise 4 (max 5 points)\n",
    "\n",
    "Consider the `x` and `y` columns as the coordinates in a plane. Add a column `avg_coll_dist` to the data with the average distance of each butterfly with respect to the other butterflies collected in the same date (year and month).\n",
    "\n",
    "To get the full marks avoid the use of explicit loops."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "ab704e78",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "def avg_coll_dist(group):\n",
    "    x = group['x'].values\n",
    "    y = group['y'].values\n",
    "    dist_vec = np.vectorize(distance)\n",
    "    dists = dist_vec(x, y, x[:, np.newaxis], y[:, np.newaxis])\n",
    "    dists = np.triu(dists, k=1) # exclude diagonal and lower triangular elements\n",
    "    return dists.mean()\n",
    "\n",
    "df['avg_coll_dist'] = df.groupby(['year', 'month']).apply(avg_coll_dist).reset_index(drop=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bbe6a088",
   "metadata": {},
   "source": [
    "Defining a function avg_coll_dist that takes a group as an argument. This function is meant to calculate the average collision distance for a group of data.\n",
    "\n",
    "Initialize two numpy arrays x and y using the values of the 'x' and 'y' columns of the input group.\n",
    "Define a vectorized version of the distance function using np.vectorize.\n",
    "Calculate the pairwise distances between the elements of the arrays x and y using dists = dist_vec(x, y, x[:, np.newaxis], y[:, np.newaxis]).\n",
    "Exclude the diagonal and lower triangular elements of the dists array using dists = np.triu(dists, k=1).\n",
    "Return the mean of the resulting dists array using return dists.mean().\n",
    "Add a new column 'avg_coll_dist' to the df DataFrame by grouping the data based on the values of the 'year' and 'month' columns, applying the avg_coll_dist function to each group, and assigning the result to the new column.\n",
    "\n",
    "\n",
    "Excluding the diagonal and lower triangular elements of the dists array is necessary because the pairwise distances between each element of the arrays x and y are calculated twice - once when considering each element of x as the starting point and each element of y as the end point, and again when considering each element of y as the starting point and each element of x as the end point.\n",
    "\n",
    "Since the distance between two points is symmetrical (the distance from point A to point B is the same as the distance from point B to point A), these duplicate calculations result in a symmetrical matrix with redundant information. To avoid counting this information twice, you use np.triu(dists, k=1) to only keep the upper triangular elements of the dists matrix and exclude the diagonal and lower triangular elements. This eliminates the redundant information and ensures that each pairwise distance is only counted once."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1f3752a",
   "metadata": {},
   "source": [
    "### Exercise 5 (max 3 points)\n",
    "\n",
    "Print the mean `avg_coll_dist` for each date (month, year)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "ffe05c46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "year  month    \n",
      "2017  July         6724.795775\n",
      "      June         6724.795775\n",
      "      May          6646.123192\n",
      "      September    6724.795775\n",
      "2018  July         6724.795775\n",
      "      June         6724.795775\n",
      "      May          6646.123192\n",
      "      September    6724.795775\n",
      "2019  July         6724.795775\n",
      "      June         6724.795775\n",
      "      May          6724.795775\n",
      "      September    6724.795775\n",
      "Name: avg_coll_dist, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "mean_coll_dist = df.groupby(['year', 'month'])['avg_coll_dist'].mean()\n",
    "print(mean_coll_dist)\n",
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d9164474",
   "metadata": {},
   "source": [
    "### Exercise 6 (max 3 points)\n",
    "\n",
    "Plot a histogram with the density of `avg_coll_dist`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "cd0dec11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(df['avg_coll_dist'], bins=20, density=True, color='blue', alpha=0.5)\n",
    "plt.xlabel('Avg. Collision Distance')\n",
    "plt.ylabel('Density')\n",
    "plt.title('Histogram of Average Collision Distance')\n",
    "plt.show()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9b81b8d",
   "metadata": {},
   "source": [
    "### Exercise 7 (max 3 points)\n",
    "\n",
    "Plot together, using two different colors, the histogram with the density of `avg_coll_dist` for `organic` and non-`organic` butterflies.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ea70d77",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Replace NaN values with the mean of the available values\n",
    "avg_coll_dist_mean = np.mean(avg_coll_dist[~np.isnan(avg_coll_dist)])\n",
    "avg_coll_dist[np.isnan(avg_coll_dist)] = avg_coll_dist_mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "4c48a395",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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hg0bxsCBAxkwYACDBg2ie/fubLjhhmy55ZYMGDCAddZZhx133HGJYrj88ssZMWIEN9xwAx06dOCaa65h++23b85qNrtcn1ksqQMwHtgA+HVE/KRs/vPAnhFRm8ZfAbaNiPfKyo0ARgCsu+66W73xRoPPVzCzVvLCCy+wySabtHYYVVuwYAGff/45nTt35pVXXmG33XZj2rRpud5v0FIqfRaSxkdETaXyuV41FBELgC0krQaMlrRpRDy/FMu5DrgOoKamxkcLZrbMPvroI772ta/x+eefExFcffXV7SIJLI0WuXw0Ij6Q9BiwJ1CaCN4G1gFqJXUEugOzWiImMyu2bt26+bG3SZ5XDfVKRwJI6gLsDrxYVmwMMCwNfxt41OcHzMxaVp5HBL2Bm9N5ghWAuyLifknnAuMiYgxwA3CrpOnA+8AhOcZjZmYV5HnV0GRgywrTzy4Z/gQ4KK8YzMysab6z2Mys4NzXkJnloxX6oZbEqaeeyq9+9SsALr74YubNm8fIZopl7NixnHrqqcyZMweAU089lREjRjTLshtz9tln89WvfpXddtstl+U7EZhZu7HSSitx77338tOf/pSePXs267LfffddDjvsMO677z6GDBnCe++9x9e//nX69OnD3nvvvVjZ+fPn07Fj8/28nnvuuc22rErcNGRm7UbHjh0ZMWIEl1566Rfmvf766+y6665svvnmDB06lDfffBOA4cOHc9JJJ7HDDjuw/vrrc/fdd1dc9q9//WuGDx/OkCFDgKwX0gsvvJDzzz+/fjk/+MEP2HbbbTn99NN55ZVX2G677dhss8342c9+RteuXQGYN28eQ4cOre/K+g9/+EN9fJtssgnHHHMMgwYNYo899uDjjz+uX3ZdXM8++yw77LADgwcPZptttmHu3LnLvN2cCMysXTn++OO5/fbbv9DHz4knnsiwYcOYPHkyhx9+OCeddFL9vHfeeYexY8dy//33N/iAmUpdXNfU1DBlypT68draWp588kkuueQSTj75ZE4++WSee+45+vbtW1+mc+fOjB49mgkTJvDYY49x2mmnUXfV/Msvv8zxxx/PlClTWG211bjnnnsWW99nn33Gd77zHS6//HImTZrEww8/TJcuXZZuQ5VwIjCzdmXVVVflqKOO4oorrlhs+lNPPcVhhx0GwJFHHsnYsWPr5x1wwAGssMIKDBw4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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['organic'] = df['organic'].replace({'1': True, '0': False})\n",
    "\n",
    "df_organic = df[df['organic'] == True].dropna(subset=['avg_coll_dist'])\n",
    "df_non_organic = df[df['organic'] == False].dropna(subset=['avg_coll_dist'])\n",
    "\n",
    "plt.hist(df_organic['avg_coll_dist'], bins=10, density=True, color='blue', alpha=0.5, label='Organic')\n",
    "plt.hist(df_non_organic['avg_coll_dist'], bins=10, density=True, color='red', alpha=0.5, label='Non-Organic')\n",
    "plt.xlabel('Avg. Collision Distance')\n",
    "plt.ylabel('Density')\n",
    "plt.title('Histogram of Average Collision Distance')\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "035d1761",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<function avg_coll_dist at 0x0000029F381FDF70>\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "transect_ID\n",
       "1     6695.705225\n",
       "8     6694.096513\n",
       "10    6597.345516\n",
       "11    6597.345516\n",
       "12            NaN\n",
       "         ...     \n",
       "62            NaN\n",
       "63            NaN\n",
       "64            NaN\n",
       "66            NaN\n",
       "67            NaN\n",
       "Name: avg_coll_dist, Length: 448, dtype: float64"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print (avg_coll_dist)\n",
    "df['avg_coll_dist']\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90f6afad",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "source": [
    "### Exercise 8 (max 5 points)\n",
    "\n",
    "Consider this statistical model:\n",
    "\n",
    "\n",
    "- the coordinate `x` divided by 525000 of a butterfly in subarea 'E' is normally distributed with mean $\\mu$ and standard deviation $\\sigma$ \n",
    "- $\\mu$ is normally distributed with mean $=0$ and standard deviation $=5$\n",
    "- $\\sigma$ is exponentially distributed with $\\lambda = 1$\n",
    "\n",
    "Code this model with pymc3, sample the model, and print the summary of the resulting estimation by using `az.summary`.\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22c85b99",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pymc3 as pm\n",
    "\n",
    "\n",
    "\n",
    "# Filter data to only keep x values when subarea is 'E'\n",
    "x_data = df.loc[df['subarea'] == 'E', 'x'] / 525000\n",
    "    \n",
    "with pm.Model() as model:\n",
    "    mu_mean = 0\n",
    "    mu_sd = 5\n",
    "    sigma_lambda = 1       \n",
    "    \n",
    "    mu = pm.Normal('mu', mu=mu_mean, sd=mu_sd)\n",
    "    sigma = pm.Exponential('sigma', lam=sigma_lambda)\n",
    "    x = pm.Normal('x', mu=mu, sigma=sigma, observed=x_data)\n",
    "\n",
    "    trace = pm.sample(2000, chains=1)   \n",
    "    #trace = pm.sample(300, tune=200)\n",
    "    \n",
    "    \n",
    "az.summary(trace)"
   ]
  }
 ],
 "metadata": {
  "jupytext": {
   "text_representation": {
    "extension": ".py",
    "format_name": "light",
    "format_version": "1.5",
    "jupytext_version": "1.13.8"
   }
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
