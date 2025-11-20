from predict import predict

user_input = [5, 166, 72, 19, 175, 25.8, 0.587, 51]

result = predict(user_input)

print("\nFINAL RESULT:", result["risk"])
print("Probability:", result["probability"], "%")

print("\nDETAILED ANALYSIS:")
for line in result["analysis"]:
    print("-", line)

print("\nSUMMARY:", result["summary"])
