'''

wrapper autour du modèle DeepSpeech pour ajuster les paramètres 

'''
from deepspeech import Model
# Faudrait désinstaller deepspeech standard si on fait des tests sur 
# https://stackoverflow.com/questions/70253197/how-to-use-gpu-when-transcribing-using-deepspeech
from deepspeech-gpu import Model
# p-e import os pour etre plus modulaire avec les paths

model_file_path = 'deepspeech-0.9.3-models.pbmm'
lm_file_path = 'deepspeech-0.9.3-models.scorer'
beam_width = 500
lm_alpha = 0.93
lm_beta = 1.18

model = Model(model_file_path)
model.enableExternalScorer(lm_file_path)

model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)

class CDeepspeechModel(object):
    def __init__(self) -> None:
        self.model = None
        self.beam_width = None
        self.lm_alpha = None
        self.lm_beta = None
    
    def initialize_model(self, pbmm_path, scorer_path, Beam_width=500, lm_alpha=0.93, lm_beta=1.18):
        self.model = Model(pbmm_path)
        self.model.enableExternalScorer(scorer_path)
        self.beam_width = Beam_width
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.model.setScorerAlphaBeta(self.lm_alpha, self.lm_beta)
        self.setBeamWidth(self.beam_width)