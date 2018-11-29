## Installation
[Pipenv](https://pipenv.readthedocs.io/en/latest/) automatically creates and manages a virtualenv for this project. Installation as usual:

```
$ pip install pipenv
```

To install the _project’s dependencies_:

```
$ pipenv install
```

You can spawn a shell:

```
$ pipenv shell
```

or a command installed into the virtual environment, for example:

```
$ pipenv run python cli.py --help
```


## Getting started

```
$ python cli.py --help
usage: matrix-tool [-h] [--corpus CORPUS] [--suffix SUFFIX] [--lowercase]
                   [--mfw MFW] [--mfw_pkl MFW_PKL] [--n-mfw N_MFW]
                   [--window WINDOW] [--sentences] [--output OUTPUT]
                   [--stopwords STOPWORDS] [--term TERM] [--sublinear_tf]
                   [--tfidf TFIDF]

CLI tool to process a Wikipedia dump to a word-word matrix.

optional arguments:
  -h, --help            show this help message and exit
  --corpus CORPUS       Path to corpus directory.
  --suffix SUFFIX       Suffix of the text files.
  --lowercase           Use this parameter to lowercase all letters.
  --mfw MFW             Path to JSON file with most frequent words.
  --mfw_pkl MFW_PKL     Path to pickle file with most frequent words.
  --n-mfw N_MFW         Count tokens and use the n most frequent words.
  --window WINDOW       Context window size.
  --sentences           Use sentences instead of lines.
  --output OUTPUT       Path to output directory.
  --stopwords STOPWORDS
                        Optional external stopwords list.
  --term TERM           Get top 50 nearest neighbors for this term.
  --tfidf TFIDF         Use tf-idf weighting on the word-word matrix. Allowed values are: document, global_transform.
  --sublinear_tf        Apply sublinear tf scaling, i.e. replace tf with 1 +
                        log(tf).
```

## Example
These are the top 20 nearest neighbors for the term `stadt`:

### Context for IDF (in TF-IDF)

<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Term</th>
      <th>Cosine similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>gemeinde</td>
      <td>0.477483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>marktes</td>
      <td>0.425700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kreisstadt</td>
      <td>0.423766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gelegen</td>
      <td>0.381238</td>
    </tr>
    <tr>
      <th>5</th>
      <td>stadtteil</td>
      <td>0.380550</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ortes</td>
      <td>0.380051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>stadtteils</td>
      <td>0.377508</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ansässig</td>
      <td>0.357279</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ort</td>
      <td>0.349270</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ortslage</td>
      <td>0.345554</td>
    </tr>
    <tr>
      <th>11</th>
      <td>dorf</td>
      <td>0.344673</td>
    </tr>
    <tr>
      <th>12</th>
      <td>kernstadt</td>
      <td>0.336407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ortschaft</td>
      <td>0.335691</td>
    </tr>
    <tr>
      <th>14</th>
      <td>stadtgemeinde</td>
      <td>0.332217</td>
    </tr>
    <tr>
      <th>15</th>
      <td>hof</td>
      <td>0.330435</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ortsteils</td>
      <td>0.326245</td>
    </tr>
    <tr>
      <th>17</th>
      <td>berlins</td>
      <td>0.324712</td>
    </tr>
    <tr>
      <th>18</th>
      <td>marktgemeinde</td>
      <td>0.323925</td>
    </tr>
    <tr>
      <th>19</th>
      <td>landkreise</td>
      <td>0.316252</td>
    </tr>
    <tr>
      <th>20</th>
      <td>wohnplatz</td>
      <td>0.315749</td>
    </tr>
  </tbody>
</table>

The word frequencies were determined by sliding over the entire corpus with a window of 2 tokens. The frequencies are TF-IDF weighted (document = context = window). For all vectors the cosine similarity was calculated. The corpus contained a total of 1,981,189 articles from the German Wikipedia. The runtime was about 21 hours (on one CPU).

### Articles for IDF (in TF-IDF)

#### Without sublinear TF scaling
<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Term</th>
      <th>Cosine similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>industrie</td>
      <td>0.402932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>staat</td>
      <td>0.401412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>musée</td>
      <td>0.381124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>zeitgenössische</td>
      <td>0.321478</td>
    </tr>
    <tr>
      <th>5</th>
      <td>patienten</td>
      <td>0.310556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>statue</td>
      <td>0.292731</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ersetzt</td>
      <td>0.220614</td>
    </tr>
    <tr>
      <th>8</th>
      <td>verlag</td>
      <td>0.207787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>klassische</td>
      <td>0.187874</td>
    </tr>
    <tr>
      <th>10</th>
      <td>stewart</td>
      <td>0.187797</td>
    </tr>
    <tr>
      <th>11</th>
      <td>holland</td>
      <td>0.187020</td>
    </tr>
    <tr>
      <th>12</th>
      <td>schriftstellerin</td>
      <td>0.182546</td>
    </tr>
    <tr>
      <th>13</th>
      <td>landkreis</td>
      <td>0.182546</td>
    </tr>
    <tr>
      <th>14</th>
      <td>beginnen</td>
      <td>0.174989</td>
    </tr>
    <tr>
      <th>15</th>
      <td>arabische</td>
      <td>0.162990</td>
    </tr>
    <tr>
      <th>16</th>
      <td>begründete</td>
      <td>0.161565</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gesichert</td>
      <td>0.159022</td>
    </tr>
    <tr>
      <th>18</th>
      <td>bus</td>
      <td>0.156092</td>
    </tr>
    <tr>
      <th>19</th>
      <td>öl</td>
      <td>0.143031</td>
    </tr>
    <tr>
      <th>20</th>
      <td>wende</td>
      <td>0.142508</td>
    </tr>
  </tbody>
</table>

#### With sublinear TF scaling

<table>
  <thead>
    <tr>
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ort</td>
      <td>0.476595</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dorf</td>
      <td>0.471737</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bahnhof</td>
      <td>0.457258</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bezirk</td>
      <td>0.442929</td>
    </tr>
    <tr>
      <th>5</th>
      <td>insel</td>
      <td>0.439883</td>
    </tr>
    <tr>
      <th>6</th>
      <td>allee</td>
      <td>0.424844</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ortschaft</td>
      <td>0.417953</td>
    </tr>
    <tr>
      <th>8</th>
      <td>anlage</td>
      <td>0.417635</td>
    </tr>
    <tr>
      <th>9</th>
      <td>umbenannt</td>
      <td>0.414289</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ebenfalls</td>
      <td>0.411907</td>
    </tr>
    <tr>
      <th>11</th>
      <td>burg</td>
      <td>0.408800</td>
    </tr>
    <tr>
      <th>12</th>
      <td>gelegen</td>
      <td>0.407507</td>
    </tr>
    <tr>
      <th>13</th>
      <td>stadtteils</td>
      <td>0.403053</td>
    </tr>
    <tr>
      <th>14</th>
      <td>straße</td>
      <td>0.401647</td>
    </tr>
    <tr>
      <th>15</th>
      <td>kernstadt</td>
      <td>0.400973</td>
    </tr>
    <tr>
      <th>16</th>
      <td>orts</td>
      <td>0.398161</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ortslage</td>
      <td>0.395894</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ansässig</td>
      <td>0.388783</td>
    </tr>
    <tr>
      <th>19</th>
      <td>hütte</td>
      <td>0.383289</td>
    </tr>
    <tr>
      <th>20</th>
      <td>stadtgemeinde</td>
      <td>0.378792</td>
    </tr>
  </tbody>
</table>
