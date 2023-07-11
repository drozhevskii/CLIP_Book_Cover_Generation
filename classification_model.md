## Classification model

This step consisted of researching the best classification technique. The models tested included: Logistic Regression, Naive Bayes, and Random Forest. After performing cross-cross validation on all three, the best configurations were chosen and are present in the table below.

<img src="images/table_models_comp718.png">

From the table above, the model with the highest performance is the Naive Bayes. The low accuracy on all the models indicates that there is a weak correlation between a book’s title and its category, which makes sense in real life. Some books’ titles do not give any clues about the category they belong to. However, since it is a hard task even for a human brain, we believe that this approximation is still enough to make a good guess about a book’s category based on its title. We will use the pre-trained Naive Bayes model to make category predictions for the book title prompts in the second dataset used for fine-tuning the model.

The full code is here.

### Exploring classification models to find the best one:

### Libraries:

Install PySpark:
```
%%bash
pip install pyspark &> /dev/null
```

Import statements here and create spark and sparkcontext objects:
```
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import Row
import numpy as np
import pandas as pd
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

sqlContext = SQLContext(sc)
```
```
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import re

from pyspark.sql.functions import *
```

### Data

Import the data and filter out unnecessary columns:
```
datasource = spark.read.format("csv") \
                  .load('/content/book32-listing.csv', delimiter=',', encoding='ISO-8859-1')
```
```
newColumns = ['ASIN', 'FILENAME', 'IMAGE_URL', 'TITLE', 'AUTHOR', 'CATEGORY_ID', 'CATEGORY']
df = datasource.toDF(*newColumns)
df.printSchema()
```

Drop all the NA values from the dataset.
Add a DESCR column which combines author and title to make the best descriptive parameter for book title.
```
df_noNull = df.na.drop()

df_noNull = df_noNull.withColumn("DESCR", concat(df_noNull.TITLE,df_noNull.AUTHOR))
df2 = df_noNull.select(df_noNull.columns[3:8])
df2.show(5)
```
```
+--------------------+------------------+-----------+---------+--------------------+
|               TITLE|            AUTHOR|CATEGORY_ID| CATEGORY|               DESCR|
+--------------------+------------------+-----------+---------+--------------------+
|Mom's Family Wall...|    Sandra Boynton|          3|Calendars|Mom's Family Wall...|
|Doug the Pug 2016...|      Doug the Pug|          3|Calendars|Doug the Pug 2016...|
|Moleskine 2016 We...|         Moleskine|          3|Calendars|Moleskine 2016 We...|
|365 Cats Color Pa...|Workman Publishing|          3|Calendars|365 Cats Color Pa...|
|Sierra Club Engag...|       Sierra Club|          3|Calendars|Sierra Club Engag...|
+--------------------+------------------+-----------+---------+--------------------+
only showing top 5 rows
```


### Pre-processing

Create a few tokenizers for every column (title, author separately) as well as for the DESCR column.
Create stopwordsRemover for Title and DESCR, since there are no stopwords in the Author column.
Create a few count vectorizers for each tokenizer.
Experimented with different values for the vocab size and minDF for the countVectorizer and 30,000 and 10 were yielding the best results.
Also, use the assembler to create a single vector combining author and title.
```
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# regular expression tokenizer
regexTokenizer1 = RegexTokenizer(inputCol="TITLE", outputCol="TITLEwords", pattern="\\W")
regexTokenizer2 = RegexTokenizer(inputCol="AUTHOR", outputCol="AUTHORwords", pattern="\\W")
regexTokenizer3 = RegexTokenizer(inputCol="DESCR", outputCol="DESCRwords", pattern="\\W")

# stop words
add_stopwords = ["and","a","the"]
stopwordsRemover = StopWordsRemover(inputCol="TITLEwords", outputCol="TITLEfiltered").setStopWords(add_stopwords)
stopwordsRemover3 = StopWordsRemover(inputCol="DESCRwords", outputCol="DESCRfiltered").setStopWords(add_stopwords)

# bag of words count
countVectors1 = CountVectorizer(inputCol="TITLEfiltered", outputCol="features1", vocabSize=30000, minDF=10)
countVectors2 = CountVectorizer(inputCol="AUTHORwords", outputCol="features2", vocabSize=30000, minDF=10)
countVectors3 = CountVectorizer(inputCol="DESCRfiltered", outputCol="features", vocabSize=30000, minDF=10)

#assembler = VectorAssembler(inputCols=["vectorizedFeatures1","vectorizedFeatures2"], outputCol='features')

assembler = VectorAssembler(inputCols=["features1","features2"], outputCol='features')
```

Create a frequency label with StringIndexer:
```
from pyspark.ml.feature import StringIndexer
label_stringIdx = StringIndexer(inputCol='CATEGORY',outputCol='label').fit(df)
```

Create two pipelines in case some models break. Cross validation on logistic regression for some reason was working with pipeline2 but not with pipeline1:
```
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler

pipeline1 = Pipeline(stages=[regexTokenizer1, regexTokenizer2, stopwordsRemover, countVectors1, countVectors2, assembler, label_stringIdx])
#pipeline2 = Pipeline(stages=[regexTokenizer3, stopwordsRemover3, countVectors3, label_stringIdx])
```

Fit the pipeline to training data:
```
pipelineFit = pipeline1.fit(df2)
dataset = pipelineFit.transform(df2)
dataset.show(5)
```
```
+--------------------+------------------+-----------+---------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----+
|               TITLE|            AUTHOR|CATEGORY_ID| CATEGORY|               DESCR|          TITLEwords|         AUTHORwords|       TITLEfiltered|           features1|           features2|            features|label|
+--------------------+------------------+-----------+---------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----+
|Mom's Family Wall...|    Sandra Boynton|          3|Calendars|Mom's Family Wall...|[mom, s, family, ...|   [sandra, boynton]|[mom, s, family, ...|(12458,[3,48,58,7...|(5456,[149,1258],...|(17914,[3,48,58,7...| 27.0|
|Doug the Pug 2016...|      Doug the Pug|          3|Calendars|Doug the Pug 2016...|[doug, the, pug, ...|    [doug, the, pug]|[doug, pug, 2016,...|(12458,[48,58,126...|(5456,[45,364],[1...|(17914,[48,58,126...| 27.0|
|Moleskine 2016 We...|         Moleskine|          3|Calendars|Moleskine 2016 We...|[moleskine, 2016,...|         [moleskine]|[moleskine, 2016,...|(12458,[58,110,14...| (5456,[1269],[1.0])|(17914,[58,110,14...| 27.0|
|365 Cats Color Pa...|Workman Publishing|          3|Calendars|365 Cats Color Pa...|[365, cats, color...|[workman, publish...|[365, cats, color...|(12458,[44,48,58,...|(5456,[25,742],[1...|(17914,[44,48,58,...| 27.0|
|Sierra Club Engag...|       Sierra Club|          3|Calendars|Sierra Club Engag...|[sierra, club, en...|      [sierra, club]|[sierra, club, en...|(12458,[48,58,103...|(5456,[1287,2337]...|(17914,[48,58,103...| 27.0|
+--------------------+------------------+-----------+---------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----+
only showing top 5 rows
```


### Partition Training & Test sets

Set seed for reproducibility:
```
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))
```
```
Training Dataset Count: 135153
Test Dataset Count: 58008
```


### Models Training and Evaluation

Logistic Regression base model using Count Vector Features:
```
lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)

predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("TITLE","AUTHOR","CATEGORY","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
```

Evaluation of Logistic Regression base model:
```
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
```
```
0.5808767914899977
```

Cross-Validation for Logistic Regression(LR):
```
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

# here i use the second pipeline since first one breaks and use single DESCR column as predictor input
pipeline2 = Pipeline(stages=[regexTokenizer3, stopwordsRemover3, countVectors3, label_stringIdx])

pipelineFit = pipeline2.fit(df2)
dataset = pipelineFit.transform(df2)
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
# i experiment with regulazation parameters since the model tends to overfit a lot
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1]) # Elastic Net Parameter (Ridge = 0)
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(trainingData)

predictions = cvModel.transform(testData)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
```
Evaluation of the LR model with best parameters:
```
0.569018235934547
```

Naive Bayes:
```
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)
predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("TITLE","AUTHOR","CATEGORY","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
```

Evaluation of Naive Bayes base model:
```
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
```
```
0.5982858622693149
```

Random Forest base model:
```
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
```

Evaluation of Random Forest base model:
```
rfModel = rf.fit(trainingData)
pred = rfModel.transform(testData)
```
```
0.538456834565456
```

Cross-Validation for Random Forest:
```
# 3-Fold Cross validation for Random Forest
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

# i was trying to make this part work but I couldnt, and my teammates did not help me unfortunately
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)

cvModel = crossval.fit(trainingData)

predictions = cvModel.transform(testData)
# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
```



