from crewai import Crew,Process
from agents import Researcher, Writer, Reader
from tasks import Research_Task, Writer_Task, Reader_Task
# Initialize the Crew with the agents and tasks
crew = Crew(
    agents=[Researcher, Writer, Reader],
    tasks=[Research_Task, Writer_Task, Reader_Task],
    process=Process.sequential
)
topic="Artifical Intelligence in Finance"
import traceback
try:
    result = crew.kickoff(inputs={"topic":topic})
    result.pretty_print()
except Exception as e:
    traceback.print_exc()
    print("Error occurred:", repr(e))
