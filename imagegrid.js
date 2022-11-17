
prompt_counts = {};
model_counts = {};
sampler_counts = {};

mark = function(prompt, model, sampler, check_id) {
    check_element = document.getElementById(check_id);
    if (check_element.className == "img_selected") {
        change = -1;
        check_element.className = "img_unselected";
    }
    else {
        change = 1;
        check_element.className = "img_selected";        
    }
    increase_count = function(counts, key) {
        if (!(key in counts)) {
            counts[key] = 0;
        }
        counts[key] += change;
    };
    increase_count(prompt_counts, prompt);
    increase_count(model_counts, model);
    increase_count(sampler_counts, sampler);

    results = document.getElementById("results");
    results.innerHTML = "";
    
    print_keys = function(counts, count_type) {
        for (const [key, count] of Object.entries(counts)) {
            results.innerHTML += count_type + " " + key + ": " + counts[key] + "<br/>\n";
        }
    };
    print_keys(prompt_counts, "prompt");
    print_keys(model_counts, "model");
    print_keys(sampler_counts, "sampler");
}