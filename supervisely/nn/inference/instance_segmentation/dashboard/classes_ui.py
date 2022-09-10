from supervisely.app.widgets import Card, Text, Select, Field, Container

classes_card = Card("Model classes", "Model predicts the following classes")

classes_layout = Container([classes_card])
