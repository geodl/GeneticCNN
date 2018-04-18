from GenNN import Model, Tracker, Util

# TODO
# Activation seperate for conv layers and dense output
#

(x_train, y_train), (x_test, y_test) = Util.prepare_data()

start_with_previous_gen = True
# Can be set very high, the progress is saved after every specimen
generations = 10

# Initialize the genetic tracker
if start_with_previous_gen:
    t = Tracker.load()
else:
    t = Tracker()

count = 0
for gen in range(generations):
    for prop in t.unscored():
        count +=1
        print("Specimen {} from generation {}".format(count, t.generation))

        model = Model(prop)
        # epochs = generation?
        model.fit(x_train[:10000], y_train[:10000], epochs=t.generation)
        score = model.evaluate(x_test, y_test)
        loss, cat_accuracy = score

        t.add_model(prop, cat_accuracy)
        t.save()

    # Print top id
    print(t.top()[1])
    # Breed next generation using the best parents from the previous generation
    t.breed_next_gen()