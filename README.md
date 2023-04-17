# Music Genre Classification (Spark ML)

## Idea
Create few examples to demonstrate regression, classification and clustering to Java developers.
Main focus is on feature extraction and creation of interesting ML pipelines.

### Genre Classification
Given part of lyric from a lyric recognize genre.

Strategy:
* Collect raw data set of lyrics (~30k sentences in total):
  * Pop
  * Country
  * Blues
  * Jazz
  * Rock
  * Reggae
  * Hiphop
  * Metal
* Create training set, i.e. label (0|1|2|3|4|5|6|7) + features
* Train logistic regression

## Build, Configure and Run

### Build
Standard build:
```
./gradlew clean build shadowJar
```
Quick build without tests:
```
./gradlew clean build shadowJar -x test
```
### Configuration
All available configuration properties are spread out via 3 files:
* application.properties - contains business logic specific stuff
* spark.properties - contains Spark specific stuff

All properties are self explanatory, but few the most important ones are listed explicitly below. 

#### Application Properties
| Name | Type | Default value | Description |
| ---- | ---- | ------------- | ----------- |
| server.port | Integer | 9090 | The port to listen for incoming HTTP requests |

#### Spark Properties
| Name | Type | Default value | Description |
| ---- | ---- | ------------- | ----------- |
| spark.master | String | spark://127.0.0.1:7077 | The URL of the Spark master. For development purposes, you can use `local[n]` that will run Spark on n threads on the local machine without connecting to a cluster. For example, `local[2]`. |
|spark.distributed-libraries | String | | Path to distributed library that should be loaded into each worker of a Spark cluster. |

#### Sample configuration for a local development environment
Create *application.properties* (for instance, in your user home directory) and override any of the described properties. 
For instance, minimum set of values that should be specified for your local environment is listed below:
```
spark.distributed-libraries=<path_to_your_repo>/spark-distributed-library/build/libs/spark-distributed-library-1.0-SNAPSHOT-all.jar

lyrics.training.set.directory.path=data/lyrics/
lyrics.model.directory.path=data/lyrics/model
```
### Run

From your favourite IDE plese run `ApplicationConfiguration` main method. 
This will use default configuration bundled in the source code. 

In order to run the application with custom configuration please add spring.config.location parameter that corresponds to directory that contains your custom *application.properties* (in our example your user home directory). Or just enumerate them explicitly, for instance:
```
spring.config.location=/Users/<your user>/application.properties
```

## Presentation and Demo

Can check out the Spark ML model prediction in the following video.

[Watch the video](https://drive.google.com/file/d/1Ou_vmNAkbeLf8k_XJLyk0Em0rUg7OWds/view?usp=share_link)

## Author

Name: [Banura Randika Perera](https://github.com/randikabanura) <br/>
Linkedin: [randika-banura](https://www.linkedin.com/in/randika-banura/) <br/>
Email: [randika.banura@gamil.com](mailto:randika.banura@gamil.com) <br/>

## Show your support

Please ⭐️ this repository if this project helped you!

## License

See [LICENSE](LICENSE) © [randikabanura](https://github.com/randikabanura/)