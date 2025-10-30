import random


def guess_number():
    print("Guess the Number Game!")
    number = random.randint(1, 100)
    attempts = 0

    while True:
        try:
            guess = int(input("Guess a number between 1 and 100: "))
            attempts += 1

            if guess < number:
                print("The number is higher!")
            elif guess > number:
                print("The number is lower!")
            else:
                print(f"Congratulations! You guessed the number in {attempts} attempts!")
                break

        except ValueError:
            print("Please enter a valid integer!")


guess_number()