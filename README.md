# TSExplanation

TSExplanation is incorporated in the framework of the engineering training at the INSA in Rennes. It has been made by a group of eight second-year IT students and goes through the whole 2018-2019 academic year.

The topics addressed are those of Machine Learning, and more specifically the classification of time series.

The result returned by a classifier never includes any explanation. The human being cannot know the reasons why the classifier has chosen this result. An explanation could provide more confidence in such a tool.

The aim of this project is to implement a tool able to explain a decision made by a time series classifier. This explanation must be clear and quite simple in order to be understandable by any user.

To make this goal easier to understand, we can take the example of a classifier that decides if a patient has a heart disease from electrocardiograms.

If a doctor uses such classifier, he would like to trust the decision made by his tool. To do so, the doctor would eventually like to be able to ask the tool to explain the classification choice (the diagnosis of the disease).
							
### Requirements

* setuptools==40.6.2
* tslearn==0.1.28.2
* pandas==0.24.2
* numpy==1.16.2
* scipy==1.2.1
* Keras==2.2.4
* matplotlib==2.1.0
* ipython==7.5.0
* PyQt5==5.12.1
* scikit_learn==0.20.3

### GUI

```
> cd GUI
> python3 MainWindow.py
```

## Authors
* [**BURIDANT Adrien**](https://github.com/insaDidi)
* [**CAM Morgane**](https://github.com/mo-cam)
* [**CHAFFIN Antoine**](https://github.com/NohTow)
* [**COUANON Yohan**](https://github.com/yocouanon)
* [**Guillou Isabelle**](https://github.com/isa-guillou)
* [**MENDES tangi**](https://github.com/tangimds)
* [**RELION Lisa**](https://github.com/lisa-relion)
* [**YASSINE Taha**](https://github.com/taha-yassine)

## Project managers
* [**GUILLEME Mael**](https://github.com/MaelG)
* [**ROZE Laurence**](https://github.com/roze35)