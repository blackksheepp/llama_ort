use ndarray::{s, Array, Array1, Array2, Array4, Axis, Dim};
use ort::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider, Session,
    SessionInputValue, TensorRTExecutionProvider, Value, ValueType,
};
use rand::thread_rng;
use rand_distr::{Distribution, WeightedIndex};
use std::error::Error;
use std::io::{self, Write};
use std::path::Path;
use tokenizers::Tokenizer;

struct ModelConfig {
    num_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Define the model path
    let model_path = "./models/model.onnx";

    // Check if the model file exists
    if !Path::new(model_path).exists() {
        return Err(Box::from(format!("Model file not found at {}", model_path)));
    }

    // Initialize the ONNX Runtime Environment and Build the ONNX session with DirectML support
    let session = Session::builder()?
        .with_execution_providers([
            // Prefer TensorRT over CUDA.
            TensorRTExecutionProvider::default().build(),
            CUDAExecutionProvider::default().build(),
            // Use DirectML on Windows if NVIDIA EPs are not available
            DirectMLExecutionProvider::default().build(),
            // Or use ANE on Apple platforms
            CoreMLExecutionProvider::default().build(),
        ])?
        .commit_from_file(model_path)?;

    // println!("Model loaded successfully from {}]\n", model_path);
    println!("Welcome to Llama-3.2-1B!\n");

    // Tokenizer loading from Hugging Face
    let tokenizer_path = "onnx-community/Llama-3.2-1B";
    let tokenizer = match Tokenizer::from_pretrained(tokenizer_path, None) {
        Ok(tokenizer) => tokenizer,
        Err(err) => panic!("Error loading tokenizer: {}", err),
    };

    // println!("Tokenizer loaded successfully from {}\n", tokenizer_path);

    // Initialize the model configuration by inspecting input tensors
    let model_config = initialize_model_config(&session)?;

    // println!(
    //     "Model Config: num_layers = {}, num_attention_heads = {}, head_dim = {}, hidden_size = {}\n\n",
    //     model_config.num_layers,
    //     model_config.num_attention_heads,
    //     model_config.head_dim,
    //     model_config.hidden_size
    // );

    loop {
        let mut prompt = String::new();

        print!("\nPrompt: ");
        io::stdout().flush().unwrap();

        std::io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to read line");

        if prompt.trim().is_empty() {
            break;
        }
        let response = match generate(
            &session,
            &tokenizer,
            &model_config,
            prompt.as_str(),
            20,
            0.7,
            0.5,
        ) {
            Ok(response) => response,
            Err(err) => panic!("Error generating response: {}", err),
        };

        if !response.is_empty() {
            if response.starts_with(prompt.to_string().as_str()) {
                let response = &response[prompt.len()..];
                println!("Llama: {}", response);
            } else {
                println!("Llama: {}", response);
            }
        }
    }

    Ok(())
}

fn initialize_model_config(session: &Session) -> Result<ModelConfig, Box<dyn Error>> {
    // Get the list of model inputs from the session
    let inputs = &session.inputs;

    let mut num_attention_heads = 0;
    let mut head_dim = 0;
    let mut num_layers = 0;

    // Iterate through the inputs to find past_key_values and architecture details
    for input in inputs {
        let input_name = input.name.clone();

        // Match on the input type to find Tensor inputs
        if let ValueType::Tensor { dimensions, .. } = &input.input_type {
            if input_name.contains("past_key_values") && input_name.contains(".key") {
                // Assuming shape format: [batch_size, num_heads, seq_length, head_dim]
                if dimensions.len() == 4 {
                    num_attention_heads = dimensions[1] as usize; // num_heads
                    head_dim = dimensions[3] as usize; // head_dim
                }
            }
        }
    }

    // Count layers based on the number of unique past_key_values
    for input in inputs {
        let input_name = input.name.clone();
        if input_name.contains("past_key_values") && input_name.contains(".key") {
            num_layers += 1;
        }
    }

    // println!(
    //     "Model Configuration: \nNum Attention Heads: {}\nHead Dim: {}\nNum Layers: {}\nHidden Size: {}",
    //     num_attention_heads, head_dim, num_layers, hidden_size
    // );

    Ok(ModelConfig {
        num_attention_heads,
        head_dim,
        num_layers,
    })
}

fn prepare_inputs<'a>(
    input_ids: &'a [u32],
    attention_mask: &'a [u32],
) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>), Box<dyn Error>> {
    let batch_size = 1;
    let seq_length = input_ids.len();

    // Make sure input tensors are properly shaped for the model
    let input_ids_array: Array2<i64> = Array2::from_shape_vec(
        (batch_size, seq_length),
        input_ids.iter().map(|&x| x as i64).collect(),
    )?;

    let attention_mask_array: Array2<i64> = Array2::from_shape_vec(
        (batch_size, seq_length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    )?;

    // Position IDs should match the sequence length
    let position_ids: Array2<i64> =
        Array2::from_shape_fn((batch_size, seq_length), |(_, j)| j as i64);

    Ok((input_ids_array, attention_mask_array, position_ids))
}

// Function to compute softmax for a 1D array
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.fold(f32::NEG_INFINITY, |max, &x| max.max(x));
    let exp_values = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp = exp_values.sum();
    exp_values / sum_exp // Normalize to get probabilities
}

// Function to compute softmax for a 2D array along the last dimension
fn softmax_2d(logits: &Array2<f32>) -> Array2<f32> {
    let axis = Axis(1); // Compute softmax along the last dimension (rows)

    // Find the maximum logit for numerical stability
    let max_logit = logits.fold(f32::NEG_INFINITY, |max, &x| max.max(x));
    // Compute e^(logit - max_logit)
    let exp_values = logits.mapv(|x| (x - max_logit).exp());
    // Sum along the specified axis
    let sum_exp = exp_values.sum_axis(axis);
    // Broadcasting the sum to normalize along the specified axis
    let sum_exp_broadcasted = sum_exp.insert_axis(axis);
    // Normalize to get probabilities
    exp_values / sum_exp_broadcasted
}

// Function to sample from a multinomial distribution
fn sample_multinomial(probs: &Array2<f32>, num_samples: usize) -> Vec<usize> {
    // Create a weighted index distribution based on the probabilities
    let weights: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
    let weighted_index = WeightedIndex::new(weights).unwrap();

    // Create a random number generator
    let mut rng = thread_rng();

    // Sample the specified number of tokens
    (0..num_samples)
        .map(|_| weighted_index.sample(&mut rng)) // Sample index
        .collect()
}

fn generate(
    session: &Session,
    tokenizer: &Tokenizer,
    model_config: &ModelConfig,
    prompt: &str,
    max_length: usize,
    temperature: f32,
    top_p: f32,
) -> Result<String, Box<dyn Error>> {
    let inputs = match tokenizer.encode(prompt, true) {
        Ok(inputs) => inputs,
        Err(err) => panic!("Error encoding prompt: {}", err),
    };
    let input_ids = inputs.get_ids();
    let mut attention_mask = inputs.get_attention_mask().to_vec();
    let mut generated_tokens = input_ids.to_vec();

    let mut past_key_values: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for _ in 0..model_config.num_layers {
        past_key_values.push((vec![], vec![]));
    }

    for _ in 0..max_length {
        let batch_size = 1;

        let (input_ids_array, attention_mask_array, position_ids) =
            match prepare_inputs(&generated_tokens, &attention_mask) {
                Ok(inputs) => inputs,
                Err(err) => panic!("Error preparing inputs: {}", err),
            };

        // print!("Step: {} ", step);
        // println!("Input shape: {:?}", input_ids_array.shape());
        // println!("Attention mask shape: {:?}", attention_mask_array.shape());
        // println!("Position IDs shape: {:?}", position_ids.shape());

        // Create the base inputs
        let mut session_inputs = ort::inputs![
            "input_ids" => input_ids_array,
            "attention_mask" => attention_mask_array,
            "position_ids" => position_ids,
        ]?;

        // For past key values, we need to match the expected dimensions
        for i in 0..model_config.num_layers {
            // Note the shape: [batch_size, num_heads, seq_len, head_dim]
            let key_shape = [
                batch_size,
                model_config.num_attention_heads,
                0,
                model_config.head_dim,
            ];

            let value_shape = key_shape;

            let key: Value<ort::TensorValueType<f32>> =
                Value::from_array(Array4::zeros(key_shape)).unwrap();
            let value: Value<ort::TensorValueType<f32>> =
                Value::from_array(Array4::zeros(value_shape)).unwrap();

            // println!("Key shape: {:?}", key.shape());
            // println!("Value shape: {:?}", value.shape());

            session_inputs.push((
                format!("past_key_values.{}.key", i).into(),
                SessionInputValue::from(key),
            ));
            session_inputs.push((
                format!("past_key_values.{}.value", i).into(),
                SessionInputValue::from(value),
            ));
        }

        let outputs = session.run(session_inputs)?;

        // Extract the output tensor from the model's output and get the logits for the next token
        let output_tensor = outputs[0].try_extract_tensor::<f32>().unwrap();
        let next_token_logits = output_tensor.slice(s![.., -1, ..]); // Get logits for the last token

        // Apply temperature scaling to the logits to control randomness
        let next_token_logits: Array<f32, Dim<[usize; 2]>> =
            next_token_logits.mapv(|x| x / temperature);

        // Create a vector of (logit, index) pairs for sorting
        let mut indexed_logits: Vec<(f32, usize)> = next_token_logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| (logit, i))
            .collect();

        // Sort logits in ascending order
        indexed_logits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Unzip the sorted logits and their corresponding indices
        let (sorted_logits, sorted_indices): (Vec<f32>, Vec<usize>) =
            indexed_logits.into_iter().unzip();

        // Convert sorted logits back into an ndarray array
        let sorted_logits = Array::from(sorted_logits);

        // Apply softmax to the sorted logits
        let sorted_logits_softmax = softmax(&sorted_logits);

        // Compute cumulative probabilities manually
        let mut cumulative_probs = Array::zeros(sorted_logits_softmax.len());
        cumulative_probs[0] = sorted_logits_softmax[0]; // Initialize the first element

        // Create cumulative sum of probabilities
        for i in 1..sorted_logits_softmax.len() {
            cumulative_probs[i] = cumulative_probs[i - 1] + sorted_logits_softmax[i];
            // Cumulative sum
        }

        // Create a boolean array indicating which indices exceed the top_p threshold
        let mut sorted_indices_to_remove: ndarray::ArrayBase<
            ndarray::OwnedRepr<bool>,
            Dim<[usize; 1]>,
        > = cumulative_probs.mapv(|x| x > top_p);
        let last_index = sorted_indices_to_remove.len() - 1;

        // Shift the boolean array to properly mark indices for removal
        for i in (1..=last_index).rev() {
            sorted_indices_to_remove[i] = sorted_indices_to_remove[i - 1];
        }

        sorted_indices_to_remove[0] = false; // Ensure the first index is not removed
        let sorted_indices_to_remove = sorted_indices_to_remove.mapv(|x| if x { 1 } else { 0 });

        // Prepare an array to hold the removal masks for the original indices
        let mut indices_to_remove: ndarray::ArrayBase<ndarray::OwnedRepr<u32>, Dim<[usize; 1]>> =
            Array::zeros(sorted_indices.len());
        for j in 0..sorted_indices.len() {
            let index = sorted_indices[j];
            indices_to_remove[index] = sorted_indices_to_remove[j]; // Mark indices to remove based on sorting
        }

        // Mask the logits, replacing those marked for removal with negative infinity
        let mut masked_logits = next_token_logits.clone();
        for (logit, &mask) in masked_logits.iter_mut().zip(indices_to_remove.iter()) {
            if mask == 1 {
                // Check if mask is 1 (indicating removal)
                *logit = f32::NEG_INFINITY; // Replace with negative infinity to remove from consideration
            }
        }

        // Apply softmax to the masked logits to get probabilities for sampling
        let probs = softmax_2d(&next_token_logits);
        let next_token = sample_multinomial(&probs, 1); // Sample the next token based on probabilities

        // Check for special tokens that indicate the end of generation
        if [128001, 128009].contains(&next_token[0]) {
            break; // Exit the loop if a stop token is generated
        }

        // Append the sampled token to the generated tokens and update the attention mask
        generated_tokens.push(next_token[0] as u32);
        attention_mask.push(1);
    }

    // Decode the generated token IDs back into text
    let generated_text = match tokenizer.decode(&generated_tokens, true) {
        Ok(text) => text,
        Err(err) => panic!("Error decoding generated text: {}", err),
    };

    Ok(generated_text)
}
