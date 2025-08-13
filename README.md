# 2_1_gauge_nn

## Dependecies

- Git
- UV

## Instructions for Users

1. Ensure that you have `uv` installed. To do so, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository by running `git clone git@github.com:alexk101/2_1_gauge_nn.git` in a terminal
3. Navigate to the project and run `uv sync` to install the environment
4. Activate the evironment by running `source .venv/bin/activate`
5. Run any of the scripts in a terminal with `python` (ex `python gauge_eqn.py`)

## Instructions for Development

1. Ensure that you have access to the repository by providing your email to the maintainer
2. Create an ssh key on your machine (you need to do this for any machine you wish to run the code on, including HPC systems) by following [these](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) instructions. If you are editing code on a remote system (HPC) create your ssh key there. If you are making changes on your local machine, make the changes there.
3. Follow all of the steps in the **Instructions for Users**
4. When you make changes to the code that you wish to contribute, you can use the command `git add -A` to add all of the changes you made, or `git add <filename>` to add changes from a specific file.
5. To commit these changes, run `git commit -m '<a message describing your changes>'`
6. Finally, to add this all to github, run `git push`

From time to time, you should also run `git pull` on your own machine to pull changes from anyone else that might have also made changes to the code, into your own branch. If you encounter a merge conflict, which is when another person has modified the same code as yourself, you can ask an LLM for help on how to resolve it.