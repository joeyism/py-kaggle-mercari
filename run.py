import pip

def install(package):
        pip.main(['install', package])

install("pyspark")

from pyspark.sql import functions, SQLContext
from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.sql.types import FloatType
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.linalg import DenseVector
import re
import itertools
from pyspark import SparkConf, SparkContext
conf = (SparkConf()
                 .setMaster("local")
                 .setAppName("mercari")
                 .set("spark.executor.memory", "5g"))
sc = SparkContext(conf = conf)


# Loading Data
sqlContext = SQLContext(sc)
df = sqlContext.read.load("data/train.tsv", format="com.databricks.spark.csv", header="true", inferSchema="true", delimiter='\t')
df_test = sqlContext.read.load("data/test.tsv", format="com.databricks.spark.csv", header="true", inferSchema="true", delimiter='\t')


# Viewing
# mean price by item_condition
#df.groupBy("item_condition_id").agg({"price": "mean"}).withColumnRenamed("avg(price)","mean_price").orderBy("mean_price").collect()
#
#
## Get all unique brand names
#join_text = functions.udf(lambda x: " ".join(set(x)))
#df.groupBy("item_condition_id").agg(functions.collect_list("brand_name").alias("brand_name")).withColumn("brand_name", join_text("brand_name")).collect()
#
## Get number of unique brands per item_condition_id
#df.groupBy("item_condition_id").agg(functions.countDistinct("brand_name")).orderBy("item_condition_id").collect()
#
## no of unique brands total
#df.select("brand_name").distinct().count()


def getAnalogy(s, model):
    qry = model.transform(s[0]) - model.transform(s[1]) - model.transform(s[2])
    res = model.findSynonyms((-1)*qry,5) # return 5 "synonyms"
    res = [x[0] for x in res]
    for k in range(0,3):
        if s[k] in res:
            res.remove(s[k])
    return res[0]

def clean_description(text):
    text = text.lower().strip()
    text = re.sub("[^0-9a-zA-Z ]", " ", text)
    text = re.sub("[  ]", " ",text)
    return text

clean_description_udf = udf(clean_description)

def group_by_words(dataframe):
    return dataframe.rdd.map(lambda x: x[0]).flatMap(lambda x: x.split(" ")).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="description_words", outputCol="word2vec_results")

### Clean Training Data
df = df.fillna({"category_name": ""})
df_test = df_test.fillna({"category_name": ""})
category_dict = dict(df.select("category_name").distinct().rdd.map(lambda r: r[0].split("/")[0]).distinct().zipWithIndex().collect())
global_category_dict = sc.broadcast(category_dict)
category_udf = udf(lambda x: global_category_dict.value[x.split("/")[0]])

def clean_data(df):
    df = df.fillna({"item_description": ""})
    df = df.withColumn("item_description", clean_description_udf("item_description"))
    df = df.withColumn("description_words", functions.split("item_description", " "))
    df = df.withColumn("id", monotonically_increasing_id())
    global category_udf
    df = df.withColumn("category_id", category_udf("category_name"))
    return df

def get_word2vec(model, description_words):
    result = model.transform(description_words)
    result = result.withColumn("id", monotonically_increasing_id())
    result = result.withColumn("x", udf(lambda v: float(v[0]), FloatType())("word2vec_results"))
    result = result.withColumn("y", udf(lambda v: float(v[1]), FloatType())("word2vec_results"))
    result = result.withColumn("z", udf(lambda v: float(v[2]), FloatType())("word2vec_results"))
    return result



df = clean_data(df)

model = word2Vec.fit(df.select("description_words"))

word2vec_results = get_word2vec(model, df.select("description_words"))

df = df.join(word2vec_results, df["id"] == word2vec_results["id"], "inner")
df = df.select(["price", "item_condition_id", "category_id", "shipping", "x", "y", "z"])
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
input_df = spark.createDataFrame(input_data, ["label", "features"])

### Clean Test Data
df_test = clean_data(df_test)
word2vec_test = get_word2vec(model, df_test.select("description_words"))
df_test = df_test.join(word2vec_test , df_test["id"] == word2vec_test ["id"], "inner")
df_test = df_test.select(["test_id", "item_condition_id", "category_id", "shipping", "x", "y", "z"])
test_data = df_test.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
input_test = spark.createDataFrame(test_data, ["test_id", "features"])


### Gradient Boosted Tree
gbt = GBTRegressor(maxIter=10)
model = gbt.fit(input_df)
prediction = model.transform(input_test)
prediction.select(["test_id", "prediction"]).coalesce(1).write.format("com.databricks.spark.csv").save("submission")
