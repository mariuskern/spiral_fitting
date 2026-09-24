# Modified in September 2026 to support configuration overrides through environment
# variables. Values are loaded from a .env file using python-dotenv and
# take precedence over corresponding values in parameters.json.
#
# Environment variables for OUTPUT_DIR, VOLUME_ZARR, and SURFACE_ZARR are
# additionally expanded using os.path.expandvars().
#
# The generated parameters.py and parameters.h files contain the effective
# configuration after applying these environment variable overrides.


import json
import os


from dotenv import load_dotenv
load_dotenv()
for key in ["OUTPUT_DIR", "VOLUME_ZARR", "SURFACE_ZARR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


def get_value(key, value):
    env_value = os.environ.get(key)

    if env_value is None:
        return value

    if isinstance(value, bool):
        return env_value.lower() in ("1", "true", "yes", "on")
    elif isinstance(value, int):
        return int(env_value)
    elif isinstance(value, float):
        return float(env_value)
    else:
        return env_value


with open("parameters.json","r",encoding="utf-8") as f:
  data = json.load(f)
  
print(data)

with open("parameters.py","w") as f:
  for (key,value) in data.items():
    for (keyc,valuec) in value.items():
      if keyc[:11]!="__comment__":
        valuec = get_value(keyc, valuec)

        if type(valuec) is str:
          f.write(keyc + '="' + str(valuec) + '"\n')
        else:
          f.write(keyc + "=" + str(valuec) + "\n")
      else:
        f.write("# " + str(valuec)+"\n")      

with open("parameters.h","w") as f:
  for (key,value) in data.items():
    for (keyc,valuec) in value.items():
      if keyc[:11]!="__comment__":
        valuec = get_value(keyc, valuec)

        if type(valuec) is str:
          f.write('#define ' + keyc + ' "' + str(valuec) + '"\n')   
        else:
          f.write("#define " + keyc + " " + str(valuec) + "\n") 
      else:
        f.write("/* " + str(valuec) + " */\n")    
        