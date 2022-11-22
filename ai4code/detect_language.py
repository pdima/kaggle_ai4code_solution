import pandas as pd
import json
import re
import os
from tqdm import tqdm
import config

# Spacy Language Detector
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector

import langid
# from langid.langid import LanguageIdentifier, model


def preprocess_text_v1(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    # return document

    return document



def detect_language_for_files(data_dir):
    nlp_model = spacy.load("en_core_web_sm")

    def get_lang_detector(nlp, name):
        return LanguageDetector(seed=42)

    # Create instance for language detection
    Language.factory("language_detector", func=get_lang_detector)
    nlp_model.add_pipe('language_detector', last=True)

    document_languages = []

    for fn in tqdm(sorted(os.listdir(data_dir))):
        if fn.endswith('.json'):
            item_id = fn[:-5]
            data = json.load(open(f'{data_dir}/{item_id}.json'))

            all_src = []

            for cell_id, source in data['source'].items():
                if data['cell_type'][cell_id] == 'markdown':
                    all_src.append(preprocess_text_v1(source))

            all_src = '\n'.join(all_src)

            doc = nlp_model(all_src[:512])
            language = doc._.language

            # print(language)

            document_languages.append([item_id, language['language'], language['score']])

    return document_languages



def detect_language_for_files_v2(data_dir):

    identifier = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)

    document_languages = []

    for fn in tqdm(sorted(os.listdir(data_dir))):
        if fn.endswith('.json'):
            item_id = fn[:-5]
            data = json.load(open(f'{data_dir}/{item_id}.json'))

            all_src = []

            for cell_id, source in data['source'].items():
                if data['cell_type'][cell_id] == 'markdown':
                    all_src.append(preprocess_text_v1(source))
                    if len(all_src) > 64:
                        break

            all_src = ' '.join(all_src)

            language = identifier.classify(all_src[:2048])

            # print(language)

            document_languages.append([item_id, language[0], language[1]])

    return document_languages


if __name__ == '__main__':
    document_languages = detect_language_for_files_v2(f'{config.DATA_DIR}/train')
    dst_fn = f'{config.DATA_DIR}/train_languages_v2.csv'
    pd.DataFrame(document_languages, columns=['item_id', 'language', 'score']).to_csv(dst_fn, index=False)
    print(dst_fn)

