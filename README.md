## CLIP_Book_Cover_Generation

![project_structure](images/ist718-prj_structure.png) ![background](images/books_cover_background.png)

This project aims to classify books into related genres and generate book covers using CLIP. We train and adopt a classification model to predict genres for the books in Pyspark, and feed the resulting dataset of book title prompts and predicted genres into CLIP to generate book covers. CLIP is a generative model that creates images based on textual descriptions. The model employs a vanilla transformer to generate embeddings and a vanilla diffusion model to produce images from those embeddings.

The project paper can be found [here](reports/).

