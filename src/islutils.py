
from ISL_Model_parameter import ISLSignPosTranslator


class ISLUtil:
    def __init__(self, base_model,translator_model_path):
        self.base_model=base_model
        self.translator_model_path=translator_model_path
        self.isl_translator_model=ISLSignPosTranslator()
