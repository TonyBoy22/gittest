'''

'''
from deepspeech import Model
# p-e import os pour etre plus modulaire avec les paths

model_file_path = 'deepspeech-0.9.3-models.pbmm'
lm_file_path = 'deepspeech-0.9.3-models.scorer'
beam_width = 500
lm_alpha = 0.93
lm_beta = 1.18

model = deepspeech.Model(model_file_path)
model.enableExternalScorer(lm_file_path)

model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)

class CDeepspeechModel(object):
    def __init__(self) -> None:
        self.model = Model.
        pass