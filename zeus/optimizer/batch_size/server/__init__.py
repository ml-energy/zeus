"""Zeus batch size optimizer server.

[Description]

Batch size optimizer is in composed of server and client. The reason for server-client architecture is that we need to maintain the states accross all trainings.
Therefore, we need a central place that is not limited by a scope of one training. 

Server's role is maintaining and updating the state of the job based on client's report.
There are three types of states.  

- Job related states: [`JobState`][zeus.optimizer.batch_size.server.job.models.JobState]
- Trial related states: [`Trial`][zeus.optimizer.batch_size.server.batch_size_state.models.Trial]
- MAB related states: [`GaussianTsArmState`][zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmState]

Client's role is letting the server know that the user started a new training and report the result of training. 


[Structure of code]

Each domain (batch_size_state and job) is composed of repository, commands, and models. 

- Repository is the lowest layer that modifies the DB. It provides CRUD operations, and performs corresponding sql operation for a given request.
- Commands are the command (collection of arguments) to use each method in repository. It is mainly used to validate the request.
- Models are used to safely perform operations on objects. All ORM objects can be converted into these models and we also have some helper models. 

In services directory, we have a single service, `ZeusService` which performs one or more operations of repositories. It performs business logics,
and provides more complicated operations to application layer. It also has commands that validates requests of using service's method.

[Hierarchy of program]

```
                            | Application layer     | Business logic | DB operation               | Storage
                            |                       |                |                            |
Client request -> Router -> | Optimizer -> Explorer | -> ZeusService | -> JobStateRepository      | <-> DB
                            |           -> Mab      |                | -> BatchSizeStateRepository|
                            |                       |                |                            |
```

[Database Transaction]

Each session represent a single transaction. When Fastapi receives the request, it creates a single session. Then, at the end of the request, it commits
every operations to Database. 

"""
