# import libraries / packages
from flask import Flask, request, Response
from docquery import document, pipeline
from paddleocr import PaddleOCR
from paddlenlp import Taskflow
import tempfile
import json
import re
import os

app = Flask(__name__)

@app.route("/automate_verify_receipt", methods=["POST"])
def automate_verify_receipt():
    # retrieve the raw byte data of image uploaded from the HTTP request
    receiptByte = request.get_data()

    try:
        # create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            # access the image content as bytes and write the image content to the temporary file
            temp_file.write(receiptByte)
            temp_file.flush()

            # get the path of the temporary file
            temp_file_path = temp_file.name

            # print the path of the temporary file
            print("Temporary file path:", temp_file_path)

            # load OCR model into memory and define the language used
            ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

            # use the OCR model to extract texts from the image
            result = ocr_model.ocr(temp_file_path, cls=True)

            # extracting detected results from the image
            result = result[0]
            texts = [line[1][0] for line in result]
            string_result = "\n*\n".join(texts)

            # use Natural Language Processing (NLP) to extract the information
            schema = ["Transaction Reference ID Number", "Recipient Beneficiary Account Number", "Total Amount"]
            ie = Taskflow("information_extraction", schema=schema, model="uie-m-base", schema_lang="en")

            # use string to extract (PaddleNLP + PaddleOCR)
            nlp_results = ie({"text": string_result})

            # retrieve all the information extraction results
            info = {}
            failedQuestions = []
            for questions in schema:
                # store results if PaddleNLP able to extract information, else store back the questions
                if nlp_results[0].get(questions) is not None:
                    info[questions] = (nlp_results[0][questions][0]['text'])
                else:
                    failedQuestions.append(questions)

            # use DocQuery to perform OCR and information extraction from the image if there PaddleNLP failed
            if failedQuestions:
                doc = document.load_document(temp_file_path)
                pipe = pipeline('document-question-answering', model="impira/layoutlm-document-qa")
                for fq in failedQuestions:
                    answer = pipe(question=fq, **doc.context)[0]["answer"]
                    if answer.strip() is not None:
                        info[fq] = answer
                    else:
                        info[fq] = "failed to obtain"

            # validate the extraction results
            payload = {}
            transaction_id = info["Transaction Reference ID Number"]
            if transaction_id:
                if transaction_id.__contains__(":"):
                    transaction_id = transaction_id.split(":")[1]

                payload["transaction_id"] = transaction_id.strip()
            else:
                return Response(json.dumps({"message": "Transaction ID Not Found"}), status=404)

            account_number = info['Recipient Beneficiary Account Number']
            if account_number and re.findall(r'\d+', account_number):
                payload["account_number"] = ''.join(re.findall(r'\d+', account_number))
            else:
                return Response(json.dumps({"message": "Account Number Not Found"}), status=404)

            transaction_amount = info["Total Amount"]
            if transaction_amount and re.findall(r'\d+\.\d+', transaction_amount):
                payload["transaction_amount"] = ''.join(re.findall(r'\d+\.\d+', transaction_amount))
            else:
                return Response(json.dumps({"message": "Transaction Amount Not Found"}), status=404)

            # print the validated extraction results
            print(payload)

    # after processing, close and delete the temporary image file
    finally:
        if temp_file is not None:
            temp_file.close()
            os.remove(temp_file_path)
        else:
            print("Failed to download the image")

    return Response(json.dumps(payload), status = 200)