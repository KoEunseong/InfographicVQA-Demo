
def get_answer(image, question, model, processor):
    
    
    # image = Image.open("/home/ges/level3-cv-productserving-cv-10/data/infographics/images/36919.jpeg")
    # question = "What percentage of Canadian internet users search for online health information not from home"
    inputs = processor(images=image, text=question, return_tensors="pt")
    # ins = processor(images = image,text=question,return_tensors='pt').to('cuda')
    predictions = model.generate(**inputs)
    pred = processor.decode(predictions[0], skip_special_tokens=True)
    return pred
