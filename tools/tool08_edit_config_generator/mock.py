"""Schema-compatible mock edit configuration."""
def run(input_data):
    return {"status":"success","tool":"edit_config_generator","result":{
        "saved_path":input_data["output_path"],"tool_calls":["mock_save_config"],"mode":"mock"}}
