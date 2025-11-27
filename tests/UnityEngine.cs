namespace UnityEngine {
    public class Debug {
        public static void Log(object message) {
            System.Console.WriteLine(message);
        }
    }
    public class MonoBehaviour {}
    public class Random {
         public static int Range(int min, int max) {
            return new System.Random().Next(min, max);
         }
    }
}
