# ChemPropsAI

Prediction of chemical properties using neural network.

[chem-prop-ai.streamlit.app](https://marekwadinger-chem-prop-ai-main-93cfrb.streamlit.app)

This project uses NN implemented by [tensorflow](https://github.com/tensorflow/tensorflow) to predict chemical properties of numerical fingerprints retrieved from atomic structures using [dscribe](https://github.com/SINGROUP/dscribe).

All wrapped up in an user-friendly web app environment of [streamlit](https://github.com/streamlit/streamlit).

## ⚡️ Quickstart

Quick way to get your hand on the ChemPropsAI service is to use our [Streamlit app](https://marekwadinger-chem-prop-ai-main-93cfrb.streamlit.app) to play around with example data.

As a quick example, we will run the online service use historical load data to fit new model and get prediction

1. To start the service, open [Streamlit app](https://marekwadinger-chem-prop-ai-main-93cfrb.streamlit.app)
2. Import the [example data](https://github.com/MarekWadinger/chem_prop_ai/blob/main/examples/data_set.xyz)
3. The service silently fits your model
4. Select descriptor (X) and property (y)
5. Submit your query
6. Download the model pressing "Download model" for later use

Note: If the app is sleeping, you can build it by pressing the big red button on screen.

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

- Feel welcome to [open an issue](https://github.com/MarekWadinger/chem_prop_ai/issues) if you think you've spotted a bug or a performance issue.


<!-- 
## 📝 License

This algorithm is free and open-source software licensed under the .
  -->
