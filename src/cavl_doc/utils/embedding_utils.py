import torch

def prepare_inputs_for_multimodal_embedding(model, tokenizer, pixel_values, question, num_patches=None):
    device = model.device

    if '<image>' not in question:
        question = '<image>\n' + question

    if num_patches is None:
        num_patches = pixel_values.shape[0]  # número de blocos/imagens

    # Criar espaço de input para os vit_embeds
    image_tokens = '<img>' + ('<IMG_CONTEXT>' * model.num_image_token * num_patches) + '</img>'
    question = question.replace('<image>', image_tokens, 1)

    inputs = tokenizer(question, return_tensors='pt').to(device)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': pixel_values.to(device),
        'image_flags': torch.ones(pixel_values.shape[0], dtype=torch.long).to(device)
    }