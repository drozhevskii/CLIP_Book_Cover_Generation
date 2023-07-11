## Fine-tuning CLIP

With the dataset of pairs of images and book titles with predicted categories, we built the pipeline to transform and feed data into a pre-trained CLIP model to fine-tune for the task of relating book cover images and book titles with categories. Since CLIP is a large model and works with visual data, we utilize [PySpark’s TorchDistributor package](https://spark.apache.org/docs/latest//api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html) to create a series of TorchDistriubutor objects and parallelize the fine-tuning (training) process of the CLIP model. It allows us to save computational resources and fine-tune the model on large-scale data.

