# 1) Call the LC QA endpoint
$qa = Invoke-RestMethod `
    -Method POST `
    -Uri "http://localhost:8000/lc/qa" `
    -ContentType "application/json" `
    -Body (@{
        question = "What are the three cerebellar peduncles and their inputs/outputs?"
    } | ConvertTo-Json)

# Optional: inspect what came back
# $qa | ConvertTo-Json -Depth 6

# 2) Build a clean body for /export/qa
$body = @{
    answer  = $qa.answer              # string
    sources = $qa.sources             # array of source dicts
}
# If sources might be null, you can force an empty array:
# $body.sources = $qa.sources ?? @()

# 3) Call the export endpoint
$exportResponse = Invoke-RestMethod `
    -Method POST `
    -Uri "http://localhost:8000/export/qa" `
    -ContentType "application/json; charset=utf-8" `
    -Body ($body | ConvertTo-Json -Depth 10)

$exportResponse

# 1) Generate the quiz
$quiz = Invoke-RestMethod `
    -Method POST `
    -Uri "http://localhost:8000/lc/quiz" `
    -ContentType "application/json" `
    -Body (@{
        topic = "cranial nerve nuclei"
        n     = 5
    } | ConvertTo-Json)

# Optional debug:
# $quiz | ConvertTo-Json -Depth 10


# 2) Build proper JSON body for export
$body = @{
    items = $quiz.items    # must be an array of objects
}

# 3) Send to /export/quiz
$exportQuiz = Invoke-RestMethod `
    -Method POST `
    -Uri "http://localhost:8000/export/quiz" `
    -ContentType "application/json; charset=utf-8" `
    -Body ($body | ConvertTo-Json -Depth 10)

$exportQuiz
