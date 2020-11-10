# Development project in Machine Learning

## Git Workflow

### Prerequisite
You need to install git, on a terminal run :
```shell 
brew install git
```

### Cloning the project
```shell
cd <path_wanted_to_dowload_project>
git clone https://github.com/thibaultspriet/developmentProjectML.git 
```

### Add your modification
You work locally on the files. Then you need to add the edits on the remote repository in order that everyone has access to them.
```shell
git add .
git commit -m "message to explain changes"
git push origin master
```

### Get modifications from other collaborator
Before edditing your functions, get the latest changes from the remote repository :
```shell 
git pull origin master
```